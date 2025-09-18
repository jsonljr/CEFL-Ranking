import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import argparse
from CFL.fedavg import Client, Server, estimate_flops
from CFL.dataset import create_non_iid_datasets, create_non_IID_datasets_strict
from CFL.model import SimpleCNN, EMNISTNet, CIFAR10Net, ResNet18
from compressor.SCA import SCACompressor

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(0)

class SCAClient(Client):
    def __init__(self, client_id, train_loader, device, lr=0.01, sparsity_fraction=0.1):
        super().__init__(client_id, train_loader, device, lr)
        self.sparsity_fraction = sparsity_fraction
        self.compressor = SCACompressor(sparsity_fraction)
        self.single_iter_flops = 0
        self.last_sca_flops = 0

    def receive_model(self, global_model):
        if self.model is None:
            self.model = copy.deepcopy(global_model).to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            self.model.load_state_dict(global_model.state_dict())
        try:
            model_cpu = copy.deepcopy(global_model).cpu()
            try:
                sample = next(iter(self.train_loader))[0]
                if isinstance(sample, (list, tuple)):
                    sample = sample[0]
                batch_size = int(sample.shape[0])
                input_size = tuple(sample.shape[1:])
            except Exception:
                input_size = (3, 32, 32) if model_cpu and hasattr(model_cpu, 'conv1') and model_cpu.conv1.in_channels == 3 else (1, 28, 28)
                batch_size = 1
            self.single_iter_flops = estimate_flops(model_cpu, input_size=input_size, batch_size=batch_size, device='cpu')
        except Exception as e:
            print(f"[WARN] estimate_flops failed: {e}")
            self.single_iter_flops = 0
        self.initial_state_dict = {k: v.clone().detach().to(self.device) for k, v in self.model.state_dict().items() if not k.startswith('_')}

    def local_train(self, epochs=1):
        if self.model is None:
            raise ValueError("模型未初始化，请先接收全局模型")
        self.model.train()
        total_iterations = 0
        start_time = time.time()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                total_iterations += 1
        end_time = time.time()
        training_time = end_time - start_time
        self.local_training_time += training_time
        flops_this_round = self.single_iter_flops * total_iterations
        self.local_flops += flops_this_round
        delta = {}
        current_state = self.model.state_dict()
        for name in self.initial_state_dict.keys():
            if name in current_state:
                delta[name] = current_state[name] - self.initial_state_dict[name]
        compressed_delta = self.compressor.compress(delta)
        sca_flops = 0
        for name, (values, indices) in compressed_delta.items():
            k = values.numel()
            n = delta[name].numel()
            sca_flops += 2 * n + 2 * k + k
        self.last_sca_flops = sca_flops
        return compressed_delta, training_time, flops_this_round, sca_flops

    def calculate_compressed_size(self, compressed_update):
        total_size = 0
        for key, value in compressed_update.items():
            if isinstance(value, tuple) and len(value) == 2:
                values, indices = value
                total_size += values.numel() * values.element_size()
                total_size += indices.numel() * indices.element_size()
            elif isinstance(value, dict) and 'values' in value and 'indices' in value:
                values = value['values']
                indices = value['indices']
                total_size += values.numel() * values.element_size()
                total_size += indices.numel() * indices.element_size()
            elif torch.is_tensor(value):
                total_size += value.numel() * value.element_size()
            else:
                try:
                    tensor_value = torch.tensor(value)
                    total_size += tensor_value.numel() * tensor_value.element_size()
                except:
                    print(f"无法计算键 {key} 的压缩大小")
        return total_size

class SCAServer(Server):
    def __init__(self, test_loader, device, num_classes=10, global_model=None, sparsity_fraction=0.1):
        super().__init__(test_loader, device, num_classes)
        self.sparsity_fraction = sparsity_fraction
        self.compressor = SCACompressor(sparsity_fraction)

    def decompress_update(self, compressed_update, device):
        decompressed = {}
        for key, value in compressed_update.items():
            if isinstance(value, tuple) and len(value) == 2:
                values, indices = value
                values, indices = values.to(device), indices.to(device)
                original_shape = self.global_model.state_dict()[key].shape
                full_update = torch.zeros(original_shape.numel(), device=device)
                full_update[indices] = values
                decompressed[key] = full_update.view(original_shape)
            else:
                decompressed[key] = value if torch.is_tensor(value) else torch.tensor(value, device=device)
        return decompressed

    def aggregate(self, client_updates):
        if not client_updates:
            return
        device = next(self.global_model.parameters()).device
        global_update = OrderedDict()
        for key in self.global_model.state_dict().keys():
            if key in client_updates[0]:
                param_shape = self.global_model.state_dict()[key].shape
                global_update[key] = torch.zeros(param_shape, device=device)
        for update in client_updates:
            decompressed_update = self.decompress_update(update, device)
            for key in decompressed_update.keys():
                if key in global_update:
                    global_update[key] += decompressed_update[key]
        for key in global_update.keys():
            global_update[key] /= len(client_updates)
        current_state = self.global_model.state_dict()
        for key in current_state.keys():
            if key in global_update:
                current_state[key] += global_update[key]
        self.global_model.load_state_dict(current_state)

    def train(self, communication_rounds=10, local_epochs=1):
        print("开始联邦学习训练...")
        start_time = time.time()
        NETWORK_SPEED = {'low': 5, 'medium': 50, 'high': 200}
        NETWORK_PROB = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        states = list(NETWORK_PROB.keys())
        probs = list(NETWORK_PROB.values())
        for round_idx in range(communication_rounds):
            print(f"\n=== 通信轮次 {round_idx + 1}/{communication_rounds} ===")
            selected_clients = np.random.choice(self.clients, size=max(1, int(len(self.clients) * 0.4)), replace=False)
            client_updates = []
            round_computation_time = 0.0
            round_flops_model = 0
            round_flops_sca = 0
            round_comm_bytes_uplink = 0
            round_comm_bytes_downlink = 0
            round_comm_time = 0.0
            client_network_state = {client.client_id: np.random.choice(states, p=probs) for client in selected_clients}
            for client in selected_clients:
                model_state = self.global_model.state_dict()
                downlink_size = sum([param.numel() * param.element_size() for param in model_state.values()])
                net_state = client_network_state[client.client_id]
                speed = NETWORK_SPEED[net_state]
                comm_time_downlink = downlink_size / (1024 ** 2) / speed
                round_comm_bytes_downlink += downlink_size
                round_comm_time += comm_time_downlink
                print(f"客户端 {client.client_id} 网络状态: {net_state} (下行速率 {speed} MB/s), 下行通信量: {downlink_size / (1024 ** 2):.6f} MB, 通信时间: {comm_time_downlink:.4f}s")
                client.receive_model(self.global_model)
                update, training_time, flops_model, flops_sca = client.local_train(epochs=local_epochs)
                client_updates.append(update)
                compressed_size = client.calculate_compressed_size(update)
                comm_time_uplink = compressed_size / (1024 ** 2) / speed
                round_comm_bytes_uplink += compressed_size
                round_comm_time += comm_time_uplink
                print(f"客户端 {client.client_id} 上行通信量: {compressed_size / (1024 ** 2):.6f} MB, 通信时间: {comm_time_uplink:.4f}s")
                round_computation_time += training_time
                round_flops_model += flops_model
                round_flops_sca += flops_sca
            round_comm_bytes = round_comm_bytes_downlink + round_comm_bytes_uplink
            round_computation_flops = round_flops_model + round_flops_sca
            round_total_time = round_computation_time + round_comm_time
            if client_updates:
                self.aggregate(client_updates)
            self.history['per_round_time'].append(round_total_time)
            self.history['communication_cost'].append(round_comm_bytes)
            self.history['computation_cost'].append(round_computation_flops)
            self.history['total_comm_bytes'] += round_comm_bytes
            self.history['total_computation_flops'] += round_computation_flops
            self.evaluate()
            print(f"本轮下行通信总量: {round_comm_bytes_downlink / (1024 ** 2):.6f} MB")
            print(f"本轮上行通信总量: {round_comm_bytes_uplink / (1024 ** 2):.6f} MB")
            print(f"本轮总通信量: {round_comm_bytes / (1024 ** 2):.6f} MB")
            print(f"本轮模型计算开销: {round_flops_model / 1e9:.4f} GFLOPs")
            print(f"本轮SCA压缩开销: {round_flops_sca} FLOPs")
            print(f"本轮总计算 FLOPs: {round_computation_flops} FLOPs")
            print(f"本轮本地训练时间: {round_computation_time:.4f}s, 通信时间: {round_comm_time:.4f}s, 总时间: {round_total_time:.4f}s")
        end_time = time.time()
        self.history['total_training_time'] = end_time - start_time
        print("联邦学习训练完成!")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "SCA_mnist10.txt")

def main():
    parser = argparse.ArgumentParser(description='联邦学习 SCA 示例')
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--communication_rounds', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--classes_per_client', type=int, default=3)
    parser.add_argument('--sparsity_fraction', type=float, default=0.1)
    parser.add_argument('--iid', action='store_true', default=False)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.iid:
        from torch.utils.data import random_split
        client_datasets = random_split(train_dataset, [len(train_dataset) // args.num_clients] * args.num_clients)
    else:
        client_datasets = create_non_iid_datasets(train_dataset, args.num_clients, args.classes_per_client)
    clients = []
    for i in range(args.num_clients):
        train_loader = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        client = SCAClient(i, train_loader, device, lr=args.lr, sparsity_fraction=args.sparsity_fraction)
        client.model = SimpleCNN().to(device)
        client.optimizer = optim.SGD(client.model.parameters(), lr=args.lr)
        clients.append(client)
    num_classes = 10
    global_model = SimpleCNN().to(device)
    server = SCAServer(test_loader, device, global_model, sparsity_fraction=args.sparsity_fraction)
    server.attach_clients(clients)
    print("初始全局模型性能:")
    server.evaluate()
    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs)
    server.plot_results()
    print("\n=== 所有轮的准确率 ===")
    for i, acc in enumerate(server.history['accuracy']):
        print(f"第{i}轮: {acc:.2f}%")

if __name__ == '__main__':
    main()
