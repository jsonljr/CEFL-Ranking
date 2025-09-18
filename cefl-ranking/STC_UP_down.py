import argparse
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import copy
from torchvision import datasets, transforms

from CFL.dataset import create_non_iid_datasets,create_non_IID_datasets_strict
from CFL.fedavg import estimate_flops

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(8 * 8 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 8 * 8 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def stc(T, hp):
    p = hp.get("p", 0.001)
    T_abs = torch.abs(T)
    n_elements = T.numel()
    n_top = max(1, int(n_elements * p))
    topk, _ = torch.topk(T_abs.flatten(), n_top)
    mean_val = topk.mean()
    out_ = torch.where(T >= topk[-1], mean_val, torch.zeros_like(T))
    out = torch.where(T <= -topk[-1], -mean_val, out_)
    flops = 3 * n_elements + n_top
    return out, flops

class STCClient:
    def __init__(self, client_id, train_loader, model, lr=0.01, up_hp={"p":0.001}):
        self.id = client_id
        self.train_loader = train_loader
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.up_hp = up_hp
        self.A = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters()}
        self.W_old = {name: param.data.clone() for name, param in self.model.named_parameters()}
        self.single_iter_flops = 0

    def receive_model(self, compressed_global_update=None, global_model=None):
        if global_model is not None:
            for name, param in self.model.named_parameters():
                param.data.copy_(global_model.state_dict()[name])
        if compressed_global_update is not None:
            for name, delta in compressed_global_update.items():
                param = dict(self.model.named_parameters())[name]
                param.data.add_(delta.to(param.device))
        self.W_old = {name: param.data.clone() for name, param in self.model.named_parameters()}

    def local_train(self, epochs=1):
        self.model.train()
        total_flops = 0
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()
                if self.single_iter_flops == 0:
                    try:
                        sample = next(iter(self.train_loader))[0]
                        input_size = tuple(sample.shape[1:])
                        batch_size = sample.shape[0]
                    except Exception:
                        input_size = (1, 28, 28)
                        batch_size = 1
                    model_cpu = copy.deepcopy(self.model).cpu()
                    self.single_iter_flops = estimate_flops(model_cpu, input_size=input_size, batch_size=batch_size,
                                                            device='cpu')
                total_flops += self.single_iter_flops

        delta_compressed = OrderedDict()
        total_comm_bytes = 0
        stc_flops = 0
        for name, param in self.model.named_parameters():
            diff = param.data - self.W_old[name]
            diff_with_error = diff + self.A[name]
            compressed,flops = stc(diff_with_error, self.up_hp)
            stc_flops += flops
            delta_compressed[name] = compressed
            self.A[name] = diff_with_error - compressed
            nz = (compressed != 0).sum().item()
            value_bytes = nz * 4
            index_bytes = nz * 4
            total_comm_bytes += value_bytes + index_bytes
        total_flops += stc_flops

        try:
            num_samples = len(self.train_loader.dataset)
        except Exception:
            num_samples = 0
            for batch in self.train_loader:
                num_samples += batch[0].shape[0]

        return delta_compressed, total_flops, total_comm_bytes, num_samples

class STCServer:
    def __init__(self, test_loader, model, down_hp={"p":0.005}):
        self.test_loader = test_loader
        self.global_model = copy.deepcopy(model).to(device)
        self.down_hp = down_hp
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

    def attach_clients(self, clients):
        self.clients = clients

    def aggregate(self, client_payloads):
        aggregated = {}
        for name in self.global_model.state_dict().keys():
            aggregated[name] = torch.mean(torch.stack([client[name] for client in client_payloads]), dim=0)

        for name, delta in aggregated.items():
            self.global_model.state_dict()[name].data.add_(delta.to(device))

        downlink = {}
        downlink_stc_flops = 0
        self.last_downlink_bytes = 0
        for name, delta in aggregated.items():
            comp, flops = stc(delta, self.down_hp)
            downlink[name] = comp
            downlink_stc_flops += flops
            nz = (comp != 0).sum().item()
            self.last_downlink_bytes += nz * 4 + nz * 4

        self.history['total_computation_flops'] += downlink_stc_flops

        return downlink, downlink_stc_flops

    def evaluate(self):
        self.global_model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                out = self.global_model(x)
                loss_sum += F.cross_entropy(out, y).item()
                pred = out.argmax(dim=1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        acc = correct / total
        self.history["accuracy"].append(acc)
        self.history["loss"].append(loss_sum/len(self.test_loader))
        print(f"Test Acc: {acc*100:.2f}%, Loss: {loss_sum/len(self.test_loader):.4f}")
        return acc

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.5):
        print("开始联邦学习训练...")
        start_time = time.time()
        for r in range(communication_rounds):
            print(f"\n=== Round {r+1}/{communication_rounds} ===")
            num_selected = max(1, int(len(self.clients)*client_frac))
            selected_clients = random.sample(self.clients, num_selected)

            client_payloads, round_flops, round_comm = [], 0, 0
            round_time = 0.0
            round_start = time.time()

            for client in selected_clients:
                client.receive_model(global_model=self.global_model)
                delta, flops, comm_bytes, num_samples = client.local_train(local_epochs)
                client_payloads.append(delta)
                round_flops += flops
                round_comm += comm_bytes
                round_time += 0

            downlink, downlink_stc_flops = self.aggregate(client_payloads)
            round_comm += self.last_downlink_bytes*num_selected
            round_flops += downlink_stc_flops
            for client in selected_clients:
                client.receive_model(compressed_global_update=downlink)

            round_end = time.time()
            round_time = round_end - round_start

            self.history["communication_cost"].append(round_comm)
            self.history["computation_cost"].append(round_flops)
            self.history["per_round_time"].append(round_time)
            self.history["total_comm_bytes"] += round_comm
            self.history["total_computation_flops"] += round_flops
            self.history["total_training_time"] += round_time

            print(f"Round Communication: {round_comm/1024:.2f} KB, Model FLOPs: {round_flops/1e6:.2f} MFLOPs, STC FLOPs: {downlink_stc_flops/1e6:.2f} MFLOPs")
            self.evaluate()

        end_time = time.time()
        self.history["total_training_time"] = end_time - start_time
        self.history["communication_rounds"] = communication_rounds
        print(f"总训练时间: {self.history['total_training_time']:.2f} 秒")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "STC_cifar10_10.txt")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--communication_rounds', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--classes_per_client', type=int, default=3)
    parser.add_argument('--iid', action='store_true', default=False)
    parser.add_argument('--up_sparsity', type=float, default=0.001)
    parser.add_argument('--down_sparsity', type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.iid:
        lengths = [len(train_dataset)//args.num_clients]*args.num_clients
        remainder = len(train_dataset)-sum(lengths)
        for i in range(remainder): lengths[i] +=1
        client_datasets = torch.utils.data.random_split(train_dataset, lengths)
    else:
        client_datasets = create_non_iid_datasets(train_dataset, args.num_clients, args.classes_per_client)

    model = CIFAR10Net().to(device)
    clients=[]
    for i in range(args.num_clients):
        loader = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        clients.append(STCClient(i, loader, model, lr=args.lr, up_hp={"p":args.up_sparsity}))

    server = STCServer(test_loader, model, down_hp={"p":args.down_sparsity})
    server.attach_clients(clients)

    print("=== 初始全局模型性能 ===")
    server.evaluate()

    start_time = time.time()
    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs)
    end_time = time.time()
    print(f"总训练时间: {end_time - start_time:.2f} 秒")

    print("\n=== 所有轮的准确率 ===")
    for i, acc in enumerate(server.history.get('accuracy', [])):
        print(f"第{i+1}轮: {acc*100:.2f}%")

if __name__=="__main__":
    main()
