
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import OrderedDict
from torch.utils.data import DataLoader
from CFL.dataset import create_non_iid_datasets,create_non_IID_datasets_strict
from CFL.fedavg import Server, Client, estimate_flops
from CFL.model import SimpleCNN,EMNISTNet,CIFAR10Net,ResNet18

class QSGDCompressor:
    def __init__(self, quantum_num=16):
        self.quantum_num = int(quantum_num)
        self.last_flops_compressor = 0

    def compress(self, tensor):
        t = tensor.detach().cpu().flatten()
        shape = tensor.size()
        numel = t.numel()
        norm = t.norm()
        if norm == 0 or torch.isnan(norm):
            quant = torch.zeros_like(t, dtype=torch.int8)
            self.last_flops_compressor = numel * 2
            return (quant, norm, shape)
        abs_t = t.abs()
        level_float = (self.quantum_num / norm) * abs_t
        previous_level = level_float.floor()
        frac = level_float - previous_level
        prob = torch.rand_like(frac)
        is_next_level = (prob < frac).to(torch.int32)
        new_level = previous_level.to(torch.int32) + is_next_level
        sign = t.sign().to(torch.int8)
        if self.quantum_num < 128:
            quant = (new_level * sign.to(torch.int32)).to(torch.int8)
        else:
            quant = (new_level * sign.to(torch.int32)).to(torch.int16)
        self.last_flops_compressor = int(numel * 8 + 100)
        return (quant, norm, shape)

    def decompress(self, compressed_tuple):
        quant, norm, shape = compressed_tuple
        out = quant.to(torch.float32)
        nval = norm.item() if isinstance(norm, torch.Tensor) else float(norm)
        scale = 0.0 if self.quantum_num == 0 else nval / float(self.quantum_num)
        deq = out * scale
        return deq.view(shape)

    def estimate_compressed_bytes(self, compressed_tuple):
        quant, norm, shape = compressed_tuple
        if quant.dtype == torch.int8:
            per = 1
        elif quant.dtype in [torch.int16, torch.float16]:
            per = 2
        else:
            per = quant.element_size()
        return int(quant.numel() * per + 4)

class QSGDClient(Client):
    def __init__(self, client_id, train_loader, device, lr=0.01, quantum_num=16):
        super().__init__(client_id, train_loader, device, lr)
        self.quantum_num = int(quantum_num)
        self.compressor = QSGDCompressor(self.quantum_num)
        self.single_iter_flops = 0
        self._initial_trainable = None
        self.local_training_time = 0.0
        self.local_flops = 0

    def receive_model(self, global_model):
        if self.model is None:
            self.model = copy.deepcopy(global_model).to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            self.model.load_state_dict(global_model.state_dict())
        self._initial_trainable = {n: p.detach().clone().to(self.device)
                                   for n, p in self.model.named_parameters() if p.requires_grad}
        try:
            model_cpu = copy.deepcopy(self.model).cpu()
            try:
                sample = next(iter(self.train_loader))[0]
                input_size = tuple(sample.shape[1:])
                batch_size = sample.shape[0]
            except Exception:
                input_size = (1, 28, 28)
                batch_size = 1
            self.single_iter_flops = estimate_flops(model_cpu, input_size=input_size, batch_size=batch_size, device='cpu')
        except Exception:
            self.single_iter_flops = 0

    def local_train(self, epochs=1, clip_grad=10.0):
        if self.model is None:
            raise RuntimeError("Call receive_model before local_train")
        self.model.to(self.device)
        self.model.train()
        start = time.time()
        total_iters = 0
        for e in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = F.cross_entropy(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                self.optimizer.step()
                total_iters += 1
        training_time = time.time() - start
        self.local_training_time += training_time
        flops_model = int(self.single_iter_flops * total_iters)
        self.local_flops += flops_model
        current_params = {n: p.detach().clone().to(self.device)
                          for n, p in self.model.named_parameters() if p.requires_grad}
        initial = self._initial_trainable or {n: torch.zeros_like(v).to(self.device) for n, v in current_params.items()}
        param_diff = OrderedDict()
        total_param_count = 0
        for n, cur in current_params.items():
            diff = cur - initial[n].to(self.device)
            param_diff[n] = diff
            total_param_count += diff.numel()
        compressed_update = OrderedDict()
        qsgd_flops = 0
        for n, diff in param_diff.items():
            compressed = self.compressor.compress(diff.detach())
            compressed_update[n] = compressed
            qsgd_flops += getattr(self.compressor, 'last_flops_compressor', 0)
        total_flops = flops_model + qsgd_flops
        self.local_flops += qsgd_flops
        try:
            num_samples = len(self.train_loader.dataset)
        except Exception:
            num_samples = 0
            for b in self.train_loader:
                num_samples += b[0].shape[0]
        return compressed_update, training_time, total_flops, num_samples

    def estimate_uplink_bytes(self, compressed_update):
        total = 0
        for n, comp in compressed_update.items():
            total += self.compressor.estimate_compressed_bytes(comp)
        return total

class QSGDServer(Server):
    def __init__(self, test_loader, device, num_classes=10, global_model=None, keep_quantum=16):
        super().__init__(test_loader, device, num_classes)
        self.keep_quantum = keep_quantum
        self.total_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        self.model_size_bytes = int(self.total_params * 4)

    def decompress_and_dense(self, compressed_update):
        device = next(self.global_model.parameters()).device
        decompressed = OrderedDict()
        compressor = QSGDCompressor(self.keep_quantum)
        for n, comp in compressed_update.items():
            deq = compressor.decompress(comp)
            decompressed[n] = deq.to(device)
        return decompressed

    def aggregate(self, client_payloads):
        if not client_payloads:
            return
        updates, ns = zip(*client_payloads)
        total = sum(ns)
        weights = [n/total if total>0 else 1.0/len(ns) for n in ns]
        first_decomp = self.decompress_and_dense(updates[0])
        avg = {k: torch.zeros_like(v, device=next(self.global_model.parameters()).device) for k, v in first_decomp.items()}
        for upd, w in zip(updates, weights):
            decomp = self.decompress_and_dense(upd)
            for k in avg.keys():
                if k in decomp:
                    avg[k] += decomp[k] * w
        with torch.no_grad():
            for name, p in self.global_model.named_parameters():
                if name in avg:
                    p.add_(avg[name].to(p.device).to(p.dtype))

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.4):
        print("Starting FedAvg with QSGD and simulated network...")
        start_all = time.time()
        NETWORK_SPEED = {'low': 5, 'medium': 50, 'high': 200}
        NETWORK_PROB = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        states = list(NETWORK_PROB.keys())
        probs = list(NETWORK_PROB.values())
        for r in range(communication_rounds):
            round_start = time.time()
            num_selected = max(1, int(len(self.clients) * client_frac))
            idx = np.random.choice(len(self.clients), size=num_selected, replace=False)
            selected_clients = [self.clients[i] for i in idx]
            print(f"\n=== Round {r + 1}, selected clients: {[c.client_id for c in selected_clients]} ===")
            client_payloads = []
            round_time = 0.0
            round_flops_model = 0
            round_flops_comp = 0
            round_uplink = 0
            round_comm_bytes = 0
            round_comm_time = 0.0
            client_network_state = {c.client_id: np.random.choice(states, p=probs) for c in selected_clients}
            downlink_bytes = num_selected * self.model_size_bytes
            for c in selected_clients:
                net_state = client_network_state[c.client_id]
                speed = NETWORK_SPEED[net_state]
                c.receive_model(self.global_model)
                comp_upd, ttime, flops, ns = c.local_train(epochs=local_epochs)
                client_payloads.append((comp_upd, ns))
                round_time += ttime
                uplink_bytes_client = c.estimate_uplink_bytes(comp_upd)
                comm_bytes = self.model_size_bytes + uplink_bytes_client
                comm_time = comm_bytes / (1024 ** 2) / speed
                round_comm_bytes += comm_bytes
                round_comm_time += comm_time
                print(f"Client {c.client_id} network: {net_state}, speed: {speed} MB/s, "
                      f"down+up: {comm_bytes / 1024 ** 2:.2f} MB, comm_time: {comm_time:.2f}s")
                comp_flops = getattr(c.compressor, 'last_flops_compressor', 0)
                round_flops_comp += comp_flops
                round_flops_model += max(0, flops - comp_flops)
                round_uplink += uplink_bytes_client
            if client_payloads:
                self.aggregate(client_payloads)
            round_total_time = round_time + round_comm_time
            self.history['communication_cost'].append(int(round_comm_bytes))
            self.history['computation_cost'].append(int(round_flops_model + round_flops_comp))
            self.history['per_round_time'].append(round_total_time)
            self.history['total_comm_bytes'] += int(round_comm_bytes)
            self.history['total_computation_flops'] += int(round_flops_model + round_flops_comp)
            self.evaluate()
            print(f"[Round {r + 1}] total comm: {round_comm_bytes / (1024 ** 2):.4f} MB, "
                  f"model FLOPs: {round_flops_model / 1e9:.4f} GFLOPs, "
                  f"comp FLOPs: {round_flops_comp}, "
                  f"train+comm time: {round_total_time:.2f}s")
        self.history['total_training_time'] = time.time() - start_all
        print("Training finished")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "QSGDcom_mnist20.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--communication_rounds', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--quantum_num', type=int, default=8)
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--classes_per_client', type=int, default=3)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
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
        loader = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        c = QSGDClient(i, loader, device, lr=args.lr, quantum_num=args.quantum_num)
        c.model = SimpleCNN().to(device)
        c.optimizer = optim.SGD(c.model.parameters(), lr=args.lr)
        clients.append(c)
    num_classes=10
    global_model = SimpleCNN().to(device)
    server = QSGDServer(test_loader, device, global_model, keep_quantum=args.quantum_num)
    server.attach_clients(clients)
    print("Initial evaluation:")
    server.evaluate()
    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs, client_frac=0.4)

if __name__ == "__main__":
    main()
