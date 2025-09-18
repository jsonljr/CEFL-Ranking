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
from torch.utils.data import DataLoader ,Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CFL.model import SimpleCNN
from CFL.dataset import create_non_iid_datasets

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
        if isinstance(m, nn.Conv2d):
            batch = o.size(0)
            out_ch = o.size(1)
            h_out, w_out = o.size(2), o.size(3)
            k_h, k_w = m.kernel_size
            in_ch = m.in_channels
            groups = m.groups
            filters_per_channel = out_ch // groups
            conv_per_position_flops = k_h * k_w * in_ch * filters_per_channel
            flops = batch * h_out * w_out * conv_per_position_flops * 2
            total_flops += flops
            if m.bias is not None:
                total_flops += batch * h_out * w_out * out_ch
        elif isinstance(m, nn.Linear):
            batch = o.size(0)
            in_f = m.in_features
            out_f = m.out_features
            flops = batch * in_f * out_f * 2
            total_flops += flops

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count_layer_flops))

    example_input = torch.randn(batch_size, *input_size).to(device)
    with torch.no_grad():
        _ = model(example_input)

    for h in hooks:
        h.remove()

    return total_flops * 3

class LotteryFLClient:
    def __init__(self, client_id, train_loader, val_loader, device, lr=0.01,
                 target_pruning_rate=0.8, pruning_rate=0.2, acc_threshold=0.5):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.target_pruning_rate = target_pruning_rate
        self.pruning_rate = pruning_rate
        self.acc_threshold = acc_threshold
        self.current_pruning_rate = 0.0
        self.model = None
        self.optimizer = None
        self.mask = None
        self.initial_params = None
        self.local_training_time = 0.0
        self.local_flops = 0
        self.single_iter_flops = 0

    def receive_model(self, global_model, global_mask=None):
        if self.model is None:
            self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(global_model.state_dict())
        if global_mask is None:
            self.mask = {name: torch.ones_like(param, dtype=torch.bool, device=self.device)
                        for name, param in self.model.named_parameters()}
        else:
            self.mask = {name: mask.clone().to(self.device) for name, mask in global_mask.items()}
        self.apply_mask()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.initial_params = {n: p.detach().clone().to(self.device)
                              for n, p in self.model.named_parameters() if p.requires_grad}
        try:
            model_cpu = copy.deepcopy(self.model).cpu()
            try:
                sample = next(iter(self.train_loader))[0]
                input_size = tuple(sample.shape[1:])
            except Exception:
                input_size = (1, 28, 28)
            self.single_iter_flops = estimate_flops(model_cpu, input_size=input_size,
                                                   batch_size=sample.shape[0] if 'sample' in locals() else 1,
                                                   device='cpu')
        except Exception:
            self.single_iter_flops = 0

    def apply_mask(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask:
                    param.data *= self.mask[name].float()

    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total if total > 0 else 0

    def prune_model(self):
        total_params = 0
        remaining_params = 0
        for name, param in self.model.named_parameters():
            if name in self.mask:
                total_params += param.numel()
                remaining_params += self.mask[name].sum().item()
        current_sparsity = 1 - remaining_params / total_params
        accuracy = self.evaluate_model()
        if current_sparsity >= self.target_pruning_rate or accuracy < self.acc_threshold:
            return False
        all_params = []
        for name, param in self.model.named_parameters():
            if name in self.mask:
                flat_param = param.data.abs().view(-1)
                flat_mask = self.mask[name].view(-1)
                all_params.append(flat_param[flat_mask].cpu())
        all_params = torch.cat(all_params)
        prune_count = int(total_params * self.pruning_rate)
        if prune_count <= 0:
            return False
        threshold = torch.topk(all_params, prune_count, largest=False).values.max().item()
        for name, param in self.model.named_parameters():
            if name in self.mask:
                abs_param = param.data.abs()
                new_mask = (abs_param > threshold) & self.mask[name]
                self.mask[name] = new_mask
        self.apply_mask()
        remaining_params = sum(self.mask[name].sum().item() for name in self.mask)
        self.current_pruning_rate = 1 - remaining_params / total_params
        print(f"[Client {self.client_id}] 剪枝完成: acc={accuracy:.2f}, 当前稀疏度={self.current_pruning_rate:.2%}")
        return True

    def reset_to_initial(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.initial_params:
                    param.data.copy_(self.initial_params[name])
        self.apply_mask()

    def local_train(self, epochs=1):
        if self.model is None:
            raise ValueError("模型未初始化，请先接收全局模型")
        self.model.to(self.device)
        self.model.train()
        accuracy = self.evaluate_model()
        if accuracy > self.acc_threshold and self.current_pruning_rate < self.target_pruning_rate:
            print(f"[Client {self.client_id}] 满足剪枝条件: acc={accuracy:.2f}, 当前稀疏度={self.current_pruning_rate:.2%}")
            self.prune_model()
            self.reset_to_initial()
        total_iterations = 0
        start_time = time.time()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in self.mask:
                            param.grad *= self.mask[name].float()
                self.optimizer.step()
                total_iterations += 1
        end_time = time.time()
        training_time = end_time - start_time
        self.local_training_time += training_time
        flops_this_round = int(self.single_iter_flops * total_iterations)
        self.local_flops += flops_this_round
        current_params = {n: p.detach().clone().to(self.device)
                         for n, p in self.model.named_parameters() if p.requires_grad}
        delta = OrderedDict()
        for n, cur in current_params.items():
            if n in self.mask:
                masked_delta = torch.zeros_like(cur)
                masked_delta[self.mask[n]] = cur[self.mask[n]] - self.initial_params[n][self.mask[n]]
                delta[n] = masked_delta
            else:
                delta[n] = cur - self.initial_params[n]
        try:
            num_samples = len(self.train_loader.dataset)
        except Exception:
            num_samples = 0
            for batch in self.train_loader:
                num_samples += batch[0].shape[0]
        return delta, self.mask, training_time, flops_this_round, num_samples

class LotteryFLServer:
    def __init__(self, test_loader, device, num_classes=10):
        self.global_model = SimpleCNN(num_classes).to(device)
        self.test_loader = test_loader
        self.device = device
        self.clients = []
        self.client_masks = {}
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
        self.model_size_bytes = sum(p.numel() * p.element_size()
                                   for p in self.global_model.parameters() if p.requires_grad)

    def attach_clients(self, clients):
        self.clients.extend(clients)
        for client in clients:
            self.client_masks[client.client_id] = None

    def aggregate(self, client_payloads):
        if not client_payloads:
            return
        updates, masks, ns = zip(*client_payloads)
        total_samples = sum(ns)
        weights = [n / total_samples for n in ns] if total_samples > 0 else [1.0 / len(ns)] * len(ns)
        overlap_mask = {}
        for name, param in self.global_model.named_parameters():
            if name in masks[0]:
                overlap_mask[name] = torch.zeros_like(param, dtype=torch.bool, device=self.device)
                count_mask = torch.zeros_like(param, dtype=torch.int, device=self.device)
                for mask in masks:
                    count_mask += mask[name].int()
                overlap_mask[name] = count_mask >= 2
        global_update = {n: torch.zeros_like(p, device=self.device)
                        for n, p in self.global_model.named_parameters() if p.requires_grad}
        for upd, w in zip(updates, weights):
            for name in global_update.keys():
                if name in upd and name in overlap_mask:
                    masked_update = upd[name] * overlap_mask[name].float()
                    global_update[name] += masked_update.to(self.device) * w
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in global_update:
                    param.add_(global_update[name])

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

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.4):
        print("开始LotteryFL训练...")
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
                client_mask = self.client_masks[client.client_id]
                client.receive_model(self.global_model, client_mask)
                downlink_bytes += self.model_size_bytes
                if client_mask is not None:
                    mask_size_bytes = sum(mask.numel() // 8 for mask in client_mask.values())
                    downlink_bytes += mask_size_bytes
            for client in selected_clients:
                update, mask, training_time, flops, num_samples = client.local_train(epochs=local_epochs)
                client_payloads.append((update, mask, num_samples))
                self.client_masks[client.client_id] = mask
                round_computation_time += training_time
                round_computation_flops += flops
                if mask is not None:
                    sparse_param_count = sum(m.sum().item() for m in mask.values())
                    uplink_bytes += sparse_param_count * 4
                else:
                    uplink_bytes += self.model_size_bytes
            round_comm_bytes = downlink_bytes + uplink_bytes
            if client_payloads:
                self.aggregate(client_payloads)
            self.history['communication_cost'].append(round_comm_bytes)
            self.history['computation_cost'].append(round_computation_flops)
            self.history['per_round_time'].append(round_computation_time)
            self.history['total_comm_bytes'] += round_comm_bytes
            self.history['total_computation_flops'] += round_computation_flops
            self.evaluate()
            print(f"本轮通信开销: {round_comm_bytes / (1024**2):.2f} MB")
            print(f"本轮计算开销: {round_computation_flops / 1e9:.2f} GFLOPs")
            print(f"本轮计算时间: {round_computation_time:.2f} 秒")
            total_params = 0
            remaining_params = 0
            for mask in self.client_masks.values():
                if mask is not None:
                    for m in mask.values():
                        total_params += m.numel()
                        remaining_params += m.sum().item()
            if total_params > 0:
                avg_sparsity = 1 - remaining_params / total_params
                print(f"平均稀疏度: {avg_sparsity:.2%}")
        end_time = time.time()
        self.history['total_training_time'] = end_time - start_time
        print("LotteryFL训练完成!")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "Lottery-mnist10.txt")

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
    parser = argparse.ArgumentParser(description='LotteryFL实现')
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--communication_rounds', type=int, default=20, help='通信轮次')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮次')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--classes_per_client', type=int, default=3, help='每个客户端的类别数')
    parser.add_argument('--iid', action='store_true', default=False, help='使用IID数据分布')
    parser.add_argument('--target_pruning_rate', type=float, default=0.8, help='目标剪枝率')
    parser.add_argument('--pruning_rate', type=float, default=0.2, help='每次剪枝的比例')
    parser.add_argument('--acc_threshold', type=float, default=0.5, help='准确率阈值')
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
        lengths = [len(train_dataset) // args.num_clients] * args.num_clients
        remainder = len(train_dataset) - sum(lengths)
        for i in range(remainder):
            lengths[i] += 1
        client_datasets = random_split(train_dataset, lengths)
    else:
        client_datasets = create_non_iid_datasets(train_dataset, args.num_clients, args.classes_per_client)
    client_train_datasets = []
    client_val_datasets = []
    for client_dataset in client_datasets:
        train_size = int(0.8 * len(client_dataset))
        val_size = len(client_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(client_dataset, [train_size, val_size])
        client_train_datasets.append(train_subset)
        client_val_datasets.append(val_subset)
    clients = []
    for i in range(args.num_clients):
        train_loader = DataLoader(client_train_datasets[i], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(client_val_datasets[i], batch_size=args.batch_size, shuffle=False)
        client = LotteryFLClient(
            i, train_loader, val_loader, device,
            lr=args.lr,
            target_pruning_rate=args.target_pruning_rate,
            pruning_rate=args.pruning_rate,
            acc_threshold=args.acc_threshold
        )
        clients.append(client)
    server = LotteryFLServer(test_loader, device)
    server.attach_clients(clients)
    print("初始全局模型性能:")
    server.evaluate()
    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs)
    print("\n=== 所有轮的准确率 ===")
    for i, acc in enumerate(server.history['accuracy']):
        print(f"第{i}轮: {acc:.2f}%")

if __name__ == '__main__':
    main()
