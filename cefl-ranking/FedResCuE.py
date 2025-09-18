import argparse
import time
import random
import math
import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CFL.model import SimpleCNN
from CFL.fedavg import create_non_iid_datasets

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

def estimate_flops(model, input_size=(1, 28, 28), batch_size=32, device='cpu'):
    model = copy.deepcopy(model).to(device)
    model.eval()
    total_flops = 0

    def count_layer_flops(m, i, o):
        nonlocal total_flops
        if isinstance(o, tuple):
            o = o[0]
        if isinstance(m, nn.Conv2d):
            batch = o.size(0)
            out_ch = o.size(1)
            h_out, w_out = o.size(2), o.size(3)
            k_h, k_w = m.kernel_size
            flops = batch * out_ch * h_out * w_out * (k_h * k_w * m.in_channels) * 2
            total_flops += flops
            if m.bias is not None:
                total_flops += batch * out_ch * h_out * w_out
        elif isinstance(m, nn.Linear):
            batch = o.size(0)
            flops = batch * m.in_features * m.out_features * 2
            total_flops += flops

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count_layer_flops))

    example_input = torch.randn(batch_size, *input_size).to(device)
    with torch.no_grad():
        output = model(example_input)

    for h in hooks:
        h.remove()
    return total_flops * 3

class FedResCuEClient:
    def __init__(self, client_id, train_loader, device, lr=0.01, capacity_ratio=1.0):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = None
        self.optimizer = None
        self.lr = lr
        self.capacity_ratio = capacity_ratio
        self.single_iter_flops = 0
        self.local_training_time = 0.0
        self.local_flops = 0

    def receive_model(self, global_model):
        if self.model is None:
            self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(global_model.state_dict())
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        try:
            model_cpu = copy.deepcopy(self.model).cpu()
            try:
                sample = next(iter(self.train_loader))[0]
                input_size = tuple(sample.shape[1:])
            except Exception:
                input_size = (1, 28, 28)
            self.single_iter_flops = estimate_flops(
                model_cpu,
                input_size=input_size,
                batch_size=sample.shape[0] if 'sample' in locals() else 1,
                device='cpu'
            )
        except Exception:
            self.single_iter_flops = 0

    def local_train(self, epochs=1):
        if self.model is None:
            raise ValueError("模型未初始化，请先接收全局模型")
        self.model.train()
        start_time = time.time()
        total_iterations = 0
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_iterations += 1
        end_time = time.time()
        training_time = end_time - start_time
        self.local_training_time += training_time
        flops_this_round = int(self.single_iter_flops * total_iterations)
        self.local_flops += flops_this_round
        current_state_dict = self.model.state_dict()
        try:
            num_samples = len(self.train_loader.dataset)
        except Exception:
            num_samples = 0
            for batch in self.train_loader:
                num_samples += batch[0].shape[0]
        return current_state_dict, training_time, flops_this_round, num_samples

class FedResCuEServer:
    def __init__(self, test_loader, device, num_classes=10):
        self.global_model = SimpleCNN(num_classes).to(device)
        self.test_loader = test_loader
        self.device = device
        self.clients = []
        self.history = {
            'accuracy': [],
            'loss': [],
            'communication_rounds': 0,
            'communication_cost': [],
            'computation_cost': [],
            'per_round_time': [],
            'total_comm_bytes': 0,
            'total_computation_flops': 0,
            'total_training_time': 0.0,
        }
        self.model_size_bytes = sum(p.numel() * p.element_size() for p in self.global_model.parameters())

    def attach_clients(self, clients):
        self.clients.extend(clients)

    def aggregate(self, client_payloads):
        if not client_payloads:
            return
        updates, ns = zip(*client_payloads)
        total_samples = sum(ns)
        weights = [n / total_samples for n in ns] if total_samples > 0 else [1.0 / len(ns)] * len(ns)
        global_update = {n: torch.zeros_like(p, device=self.device)
                        for n, p in self.global_model.named_parameters()}
        for state_dict, w in zip(updates, weights):
            for name, param in state_dict.items():
                if name in global_update:
                    global_update[name] += param.to(self.device) * w
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in global_update:
                    param.data.copy_(global_update[name])

    def evaluate(self):
        self.global_model.to(self.device)
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(test_loss)
        self.history['communication_rounds'] += 1
        print(f'通信轮次 {self.history["communication_rounds"]}: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)')
        return test_loss, accuracy

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.4, connection_error_rate=0.1):
        print("开始FedResCuE训练...")
        start_time = time.time()
        for r in range(communication_rounds):
            print(f"\n=== 通信轮次 {r+1}/{communication_rounds} ===")
            num_selected = max(1, int(len(self.clients) * client_frac))
            idx = np.random.choice(len(self.clients), size=num_selected, replace=False)
            selected_clients = [self.clients[i] for i in idx]
            client_payloads = []
            round_computation_time = 0.0
            round_computation_flops = 0
            downlink_bytes = 0
            uplink_bytes = 0
            for client in selected_clients:
                client.receive_model(self.global_model)
                downlink_bytes += self.model_size_bytes
                update, training_time, flops, num_samples = client.local_train(epochs=local_epochs)
                upload_ratio = 1.0
                if random.random() < connection_error_rate:
                    upload_ratio = max(0.25, random.uniform(0.5, 0.9))
                compressed_update = OrderedDict()
                param_keys = list(update.keys())
                num_params_to_upload = int(len(param_keys) * upload_ratio)
                for i, key in enumerate(param_keys):
                    if i < num_params_to_upload:
                        compressed_update[key] = update[key]
                client_payloads.append((compressed_update, num_samples))
                uplink_bytes += int(self.model_size_bytes * upload_ratio)
                round_computation_time += training_time
                round_computation_flops += flops
            if client_payloads:
                self.aggregate(client_payloads)
            round_comm_bytes = downlink_bytes + uplink_bytes
            self.history['communication_cost'].append(round_comm_bytes)
            self.history['computation_cost'].append(round_computation_flops)
            self.history['per_round_time'].append(round_computation_time)
            self.history['total_comm_bytes'] += round_comm_bytes
            self.history['total_computation_flops'] += round_computation_flops
            self.evaluate()
            print(f"本轮通信开销: {round_comm_bytes / (1024**2):.2f} MB")
            print(f"本轮计算开销: {round_computation_flops / 1e9:.2f} GFLOPs")
            print(f"本轮计算时间: {round_computation_time:.2f} 秒")
        self.history['total_training_time'] = time.time() - start_time
        print("FedResCuE训练完成!")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "FedResCuE_mnist10.txt")

    def print_cost_summary(self):
        total_comm_mb = self.history['total_comm_bytes'] / (1024 ** 2)
        total_computation_tflops = self.history['total_computation_flops'] / 1e12
        print("\n=== 联邦学习开销总结 ===")
        print(f"总通信轮次: {self.history['communication_rounds']}")
        print(f"总通信开销: {total_comm_mb:.2f} MB")
        print(f"总计算开销: {total_computation_tflops:.4f} TFLOPs")
        print(f"总训练时间: {self.history['total_training_time']:.2f} 秒")
        if self.history['communication_rounds'] > 0:
            print(f"平均每轮通信开销: {np.mean(self.history['communication_cost']) / (1024 ** 2):.2f} MB")
            print(f"平均每轮计算开销: {np.mean(self.history['computation_cost']) / 1e9:.2f} GFLOPs")

    def save_detailed_history_to_txt(self, history, filename):
        with open(filename, 'w') as f:
            f.write("详细训练历史记录\n")
            f.write(f"字典：{history}\n")
            f.write("=" * 60 + "\n\n")
            f.write("总体统计:\n")
            f.write("-" * 20 + "\n")
            f.write(f"总训练轮次: {history['communication_rounds']}\n")
            f.write(f"总训练时间: {history['total_training_time']:.2f} 秒\n")
            f.write(f"平均每轮时间: {history['total_training_time'] / history['communication_rounds']:.2f} 秒\n")
            f.write(f"总通信量: {history['total_comm_bytes']} 字节\n")
            f.write(f"平均每轮通信量: {history['total_comm_bytes'] / history['communication_rounds']:.0f} 字节\n")
            f.write(f"总计算量: {history['total_computation_flops']} FLOPs\n")
            f.write(f"平均每轮计算量: {history['total_computation_flops'] / history['communication_rounds']:.0f} FLOPs\n\n")
            if history['accuracy']:
                f.write("准确率统计:\n")
                f.write("-" * 20 + "\n")
                f.write(f"初始准确率: {history['accuracy'][0]:.4f}\n")
                f.write(f"最终准确率: {history['accuracy'][-1]:.4f}\n")
                f.write(f"准确率提升: {history['accuracy'][-1] - history['accuracy'][0]:.4f}\n")
                f.write(f"最高准确率: {max(history['accuracy']):.4f}\n\n")
            f.write("每轮详细数据:\n")
            f.write("-" * 20 + "\n")
            f.write("轮次\t准确率\t\t通信量(字节)\t计算量(FLOPs)\t时间(秒)\n")
            f.write("-" * 80 + "\n")
            for i in range(len(history['accuracy'])):
                acc = history['accuracy'][i]
                comm = history['communication_cost'][i] if i < len(history['communication_cost']) else 0
                comp = history['computation_cost'][i] if i < len(history['computation_cost']) else 0
                time_val = history['per_round_time'][i] if i < len(history['per_round_time']) else 0
                f.write(f"{i + 1}\t{acc:.4f}\t\t{comm}\t{comp}\t{time_val:.4f}\n")
        print(f"详细训练历史已保存到 {filename}")

def main():
    parser = argparse.ArgumentParser(description='FedResCuE联邦学习实现')
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--communication_rounds', type=int, default=20, help='通信轮次')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮次')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--classes_per_client', type=int, default=3, help='每个客户端的类别数')
    parser.add_argument('--iid', action='store_true', default=False, help='使用IID数据分布')
    parser.add_argument('--connection_error_rate', type=float, default=0.1, help='连接错误率')
    parser.add_argument('--min_capacity', type=float, default=0.25, help='最小客户端容量比例')
    parser.add_argument('--max_capacity', type=float, default=1.0, help='最大客户端容量比例')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.iid:
        lengths = [len(train_dataset) // args.num_clients] * args.num_clients
        remainder = len(train_dataset) - sum(lengths)
        for i in range(remainder):
            lengths[i] += 1
        client_datasets = random_split(train_dataset, lengths)
    else:
        client_datasets = create_non_iid_datasets(train_dataset, args.num_clients, args.classes_per_client)
    clients = []
    for i in range(args.num_clients):
        train_loader = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        capacity_ratio = random.uniform(args.min_capacity, args.max_capacity)
        client = FedResCuEClient(i, train_loader, device, lr=args.lr, capacity_ratio=capacity_ratio)
        clients.append(client)
    server = FedResCuEServer(test_loader, device)
    server.attach_clients(clients)
    print("初始全局模型性能:")
    server.evaluate()
    server.train(
        communication_rounds=args.communication_rounds,
        local_epochs=args.local_epochs,
        connection_error_rate=args.connection_error_rate
    )
    print("\n=== 所有轮的准确率 ===")
    for i, acc in enumerate(server.history['accuracy']):
        print(f"第{i}轮: {acc:.2f}%")

if __name__ == '__main__':
    main()
