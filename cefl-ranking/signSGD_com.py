
import argparse
import time
import random
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CFL.model import SimpleCNN,EMNISTNet,CIFAR10Net,ResNet18
from CFL.fedavg import Client, Server, estimate_flops
from CFL.dataset import create_non_iid_datasets,create_non_IID_datasets,create_non_IID_datasets_strict

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

class SignSGDClient(Client):
    def __init__(self, client_id, train_loader, device, lr=0.01):
        super().__init__(client_id, train_loader, device, lr)
        self.last_sign_flops = 0

    def receive_model(self, global_model):
        if self.model is None:
            self.model = copy.deepcopy(global_model).to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            self.model.load_state_dict(global_model.state_dict())
        self.initial_params = {n: p.detach().clone().cpu() for n, p in self.model.named_parameters() if p.requires_grad}
        try:
            model_cpu = copy.deepcopy(self.model).cpu()
            try:
                sample = next(iter(self.train_loader))[0]
                input_size = tuple(sample.shape[1:])
                batch_size = sample.shape[0]
            except Exception:
                input_size = (1, 28, 28)
                batch_size = 1
            self.single_iter_flops = estimate_flops(model_cpu, input_size=input_size, batch_size=batch_size)
        except Exception:
            self.single_iter_flops = 0

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
        self.local_training_time += training_time
        flops_this_round = int(self.single_iter_flops * total_iterations)
        self.local_flops += flops_this_round
        delta = OrderedDict()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                cur = p.detach().cpu()
                init = self.initial_params.get(n)
                if init is None:
                    init = torch.zeros_like(cur)
                delta[n] = cur - init
        sign_delta = OrderedDict()
        total_sign_flops = 0
        for k, v in delta.items():
            s = torch.sign(v)
            sign_delta[k] = s.to(torch.int8)
            total_sign_flops += v.numel()
        self.last_sign_flops = total_sign_flops
        try:
            num_samples = len(self.train_loader.dataset)
        except Exception:
            num_samples = 0
            for batch in self.train_loader:
                num_samples += batch[0].shape[0]
        return sign_delta, training_time, flops_this_round, num_samples

class SignSGDServer(Server):
    def __init__(self, test_loader, device, num_classes=10, global_model=None, server_lr=1.0, step_size=1e-3, use_weighted_vote=False):
        super().__init__(test_loader, device, num_classes)
        self.server_lr = server_lr
        self.step_size = step_size
        self.use_weighted_vote = use_weighted_vote
        self.sign_model_size_bytes = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)

    def aggregate(self, client_sign_payloads):
        if not client_sign_payloads:
            return
        sign_dicts, ns = zip(*client_sign_payloads)
        num_clients = len(sign_dicts)
        first_keys = list(sign_dicts[0].keys())
        accum = {k: torch.zeros_like(self.global_model.state_dict()[k], dtype=torch.float32, device=self.device) for k in first_keys}
        if self.use_weighted_vote:
            total_samples = sum(ns)
            weights = [n / total_samples if total_samples > 0 else 1.0 / num_clients for n in ns]
            for (sd, n), w in zip(client_sign_payloads, weights):
                for k, v in sd.items():
                    accum[k] += v.to(torch.float32).to(self.device) * w
        else:
            for sd in sign_dicts:
                for k, v in sd.items():
                    accum[k] += v.to(torch.float32).to(self.device)
        majority = {k: torch.sign(t) for k, t in accum.items()}
        with torch.no_grad():
            for n, p in self.global_model.named_parameters():
                if n in majority:
                    upd = (self.server_lr * self.step_size) * majority[n].to(p.device).to(p.dtype)
                    p.add_(upd)

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.4):
        start_time = time.time()
        NETWORK_SPEED = {'low': 5, 'medium': 50, 'high': 200}
        NETWORK_PROB = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        states = list(NETWORK_PROB.keys())
        probs = list(NETWORK_PROB.values())
        for r in range(communication_rounds):
            num_selected = max(1, int(len(self.clients) * client_frac))
            idx = np.random.choice(len(self.clients), size=num_selected, replace=False)
            selected_clients = [self.clients[i] for i in idx]
            client_payloads = []
            round_computation_time = 0.0
            round_computation_flops = 0
            round_flops_model = 0
            round_flops_sign = 0
            round_comm_bytes = 0
            round_comm_time = 0.0
            client_network_state = {client.client_id: np.random.choice(states, p=probs) for client in selected_clients}
            for client in selected_clients:
                comm_bytes = self.model_size_bytes + self.sign_model_size_bytes
                net_state = client_network_state[client.client_id]
                speed = NETWORK_SPEED[net_state]
                comm_time = comm_bytes / (1024 ** 2) / speed
                round_comm_bytes += comm_bytes
                round_comm_time += comm_time
                client.receive_model(self.global_model)
                sign_update, training_time, flops, num_samples = client.local_train(epochs=local_epochs)
                client_payloads.append((sign_update, num_samples))
                round_computation_time += training_time
                round_flops_model += flops
                round_flops_sign += getattr(client, 'last_sign_flops', 0)
            round_computation_flops = round_flops_model + round_flops_sign
            round_total_time = round_computation_time + round_comm_time
            if client_payloads:
                self.aggregate(client_payloads)
            self.history['communication_cost'].append(round_comm_bytes)
            self.history['computation_cost'].append(round_computation_flops)
            self.history['total_comm_bytes'] += round_comm_bytes
            self.history['total_computation_flops'] += round_computation_flops
            self.history['per_round_time'].append(round_total_time)
            self.evaluate()
        end_time = time.time()
        self.history['total_training_time'] = end_time - start_time
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "signSGDcom_mnist20.txt")

def main():
    parser = argparse.ArgumentParser(description='联邦学习: SignSGD ')
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--communication_rounds', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--classes_per_client', type=int, default=3)
    parser.add_argument('--iid', action='store_true', default=False)
    parser.add_argument('--server_lr', type=float, default=1.0)
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--use_weighted_vote', action='store_true',default=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    clients = []
    for i in range(args.num_clients):
        train_loader = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        client = SignSGDClient(i, train_loader, device, lr=args.lr)
        client.model = SimpleCNN().to(device)
        client.optimizer = optim.SGD(client.model.parameters(), lr=args.lr)
        clients.append(client)
    num_classes = 10
    global_model = SimpleCNN().to(device)
    server = SignSGDServer(test_loader, device, global_model, server_lr=args.server_lr, step_size=args.step_size,
                           use_weighted_vote=args.use_weighted_vote)
    server.attach_clients(clients)
    server.evaluate()
    start_time = time.time()
    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs)
    end_time = time.time()
    print(f"\n总训练时间: {end_time - start_time:.2f} 秒")
    for i, acc in enumerate(server.history['accuracy']):
        print(f"第{i}轮: {acc:.2f}%")

if __name__ == '__main__':
    main()
