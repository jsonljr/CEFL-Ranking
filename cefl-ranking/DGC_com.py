import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from collections import OrderedDict
from torch.utils.data import DataLoader

from CFL.dataset import create_non_iid_datasets, create_non_IID_datasets_strict
from CFL.fedavg import Server, Client, estimate_flops
from CFL.model import SimpleCNN, EMNISTNet, CIFAR10Net, ResNet18
from compressor.DGC import DGCCompressor


class DGCClient(Client):
    def __init__(self, client_id, train_loader, device, lr=0.01,
                 keep_ratio=0.01, momentum_factor=0.9, residual=True):
        super().__init__(client_id, train_loader, device, lr)
        self.compressor = DGCCompressor(keep_ratio, momentum_factor, residual)
        self.single_iter_flops = 0
        self._initial_trainable = None

    def receive_model(self, global_model):
        if self.model is None:
            self.model = copy.deepcopy(global_model).to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            self.model.load_state_dict(global_model.state_dict())
        if self.single_iter_flops == 0:
            try:
                sample = next(iter(self.train_loader))[0].to(self.device)
                self.single_iter_flops = estimate_flops(self.model, sample)
            except Exception:
                self.single_iter_flops = 0
        self._initial_trainable = {n: p.detach().clone().to(self.device)
                                   for n, p in self.model.named_parameters() if p.requires_grad}

    def local_train(self, epochs=1):
        if self.model is None:
            raise ValueError("模型未初始化，请先接收全局模型")
        self.model.to(self.device)
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
        flops_model = self.single_iter_flops * total_iterations
        current_params = {n: p.detach().clone().to(self.device)
                          for n, p in self.model.named_parameters() if p.requires_grad}
        initial = self._initial_trainable or {n: torch.zeros_like(v) for n, v in current_params.items()}
        param_diff = OrderedDict()
        num_params_total = 0
        for n, cur in current_params.items():
            diff = cur - initial[n].to(self.device)
            param_diff[n] = diff
            num_params_total += diff.numel()
        flops_compressor = num_params_total * 3
        compressed_update = self.compressor.compress(param_diff, self.model)
        try:
            num_samples = len(self.train_loader.dataset)
        except Exception:
            num_samples = 0
            for batch in self.train_loader:
                num_samples += batch[0].shape[0]
        flops_this_round = flops_model + flops_compressor
        return compressed_update, training_time, flops_this_round, num_samples


class DGCServer(Server):
    def __init__(self, test_loader, device, num_classes=10, global_model=None, keep_ratio=0.01):
        super().__init__(test_loader, device, num_classes)
        self.keep_ratio = keep_ratio
        self.total_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        self.model_size_bytes = int(self.total_params * 4)

    def _estimate_uplink_bytes_per_client(self):
        nnz = max(1, int(self.total_params * self.keep_ratio))
        index_bits = math.ceil(math.log2(self.total_params)) if self.total_params > 1 else 1
        per_item_bits = index_bits + 1
        per_item_bytes = math.ceil(per_item_bits / 8)
        uplink_bytes = nnz * per_item_bytes + 4
        return uplink_bytes

    def aggregate(self, client_payloads):
        updates, ns_list = zip(*client_payloads)
        total_samples = sum(ns_list)
        if total_samples == 0:
            weights = [1.0 / len(ns_list)] * len(ns_list)
        else:
            weights = [ns / total_samples for ns in ns_list]
        global_update = {n: torch.zeros_like(p) for n, p in self.global_model.named_parameters() if p.requires_grad}
        for upd, w in zip(updates, weights):
            for k in global_update.keys():
                if k in upd:
                    global_update[k] += upd[k] * w
        with torch.no_grad():
            for n, p in self.global_model.named_parameters():
                if p.requires_grad and n in global_update:
                    p.add_(global_update[n])

    def train(self, communication_rounds=10, local_epochs=1):
        print("开始联邦学习训练...")
        start_time = time.time()
        NETWORK_SPEED = {'low': 5, 'medium': 50, 'high': 200}
        NETWORK_PROB = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        states = list(NETWORK_PROB.keys())
        probs = list(NETWORK_PROB.values())
        for round in range(communication_rounds):
            print(f"\n=== 通信轮次 {round + 1}/{communication_rounds} ===")
            num_selected = max(1, int(len(self.clients) * 0.4))
            idx = np.random.choice(len(self.clients), size=num_selected, replace=False)
            selected_clients = [self.clients[i] for i in idx]
            client_payloads = []
            round_computation_time = 0.0
            round_flops_model = 0
            round_flops_compressor = 0
            round_comm_bytes = 0
            round_comm_time = 0.0
            client_network_state = {client.client_id: np.random.choice(states, p=probs)
                                    for client in selected_clients}
            downlink_bytes_per_client = self.model_size_bytes
            uplink_bytes_per_client = self._estimate_uplink_bytes_per_client()
            for client in selected_clients:
                net_state = client_network_state[client.client_id]
                speed = NETWORK_SPEED[net_state]
                comm_bytes = downlink_bytes_per_client + uplink_bytes_per_client
                comm_time = comm_bytes / (1024 ** 2) / speed
                round_comm_bytes += comm_bytes
                round_comm_time += comm_time
                print(f"客户端 {client.client_id} 网络状态: {net_state}, 速率: {speed} MB/s, "
                      f"通信量: {comm_bytes / (1024 ** 2):.2f} MB, 通信时间: {comm_time:.2f}s")
                client.receive_model(self.global_model)
                update, training_time, flops_this_round, num_samples = client.local_train(epochs=local_epochs)
                client_payloads.append((update, num_samples))
                round_computation_time += training_time
                flops_model = flops_this_round - getattr(client.compressor, 'last_flops_compressor', 0)
                flops_compressor = getattr(client.compressor, 'last_flops_compressor', 0)
                round_flops_model += flops_model
                round_flops_compressor += flops_compressor
            if client_payloads:
                self.aggregate(client_payloads)
            round_total_time = round_computation_time + round_comm_time
            self.history['per_round_time'].append(round_total_time)
            self.history['communication_cost'].append(round_comm_bytes)
            self.history['computation_cost'].append(round_flops_model + round_flops_compressor)
            self.history['total_comm_bytes'] += round_comm_bytes
            self.history['total_computation_flops'] += (round_flops_model + round_flops_compressor)
            self.evaluate()
            print(f"本轮下行(全模型)开销: {downlink_bytes_per_client * num_selected / (1024 ** 2):.2f} MB")
            print(f"本轮上行(估算DGC压缩)开销: {uplink_bytes_per_client * num_selected / (1024 ** 2):.4f} MB")
            print(f"本轮通信开销: {round_comm_bytes / (1024 ** 2):.4f} MB")
            print(f"本轮模型计算开销: {round_flops_model / 1e9:.2f} GFLOPs")
            print(f"本轮DGC压缩开销: {round_flops_compressor} FLOPs")
            print(f"本轮总计算开销: {(round_flops_model + round_flops_compressor) / 1e9:.2f} GFLOPs")
            print(f"本轮计算+通信总时间: {round_total_time:.2f} 秒")
        end_time = time.time()
        self.history['total_training_time'] = end_time - start_time
        print("联邦学习训练完成!")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "DGCcom_mnist10.txt")


def main_dgc():
    parser = argparse.ArgumentParser(description='联邦学习DGC实现（修正版）')
    parser.add_argument('--num_clients', type=int, default=20, help='客户端数量')
    parser.add_argument('--communication_rounds', type=int, default=200, help='通信轮次')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮次')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--classes_per_client', type=int, default=3, help='每个客户端的类别数')
    parser.add_argument('--iid', action='store_true', default=False, help='使用IID数据分布')
    parser.add_argument('--keep_ratio', type=float, default=0.01, help='上传保留比例（如0.01表示保留1%）')
    parser.add_argument('--momentum_factor', type=float, default=0.9, help='动量因子')
    parser.add_argument('--no_residual', action='store_true', default=False, help='不使用残差累积')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.iid:
        from torch.utils.data import random_split
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
        client = DGCClient(i, train_loader, device, lr=args.lr,
                           keep_ratio=args.keep_ratio,
                           momentum_factor=args.momentum_factor,
                           residual=not args.no_residual)
        client.model = SimpleCNN().to(device)
        client.optimizer = optim.SGD(client.model.parameters(), lr=args.lr)
        clients.append(client)
    num_classes = 10
    global_model = SimpleCNN().to(device)
    server = DGCServer(test_loader, device, global_model, keep_ratio=args.keep_ratio)
    server.attach_clients(clients)
    print("初始全局模型性能:")
    server.evaluate()
    start_time = time.time()
    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs)
    end_time = time.time()
    print(f"\n总训练时间: {end_time - start_time:.2f} 秒")
    print("\n=== 所有轮的准确率 ===")
    for i, acc in enumerate(server.history.get('accuracy', [])):
        print(f"第{i}轮: {acc:.2f}%")


if __name__ == '__main__':
    main_dgc()
