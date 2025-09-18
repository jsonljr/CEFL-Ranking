
import argparse
import time
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import OrderedDict
import torch.nn as nn

def set_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

set_seed(0, deterministic=False)

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

from CFL.dataset import create_non_iid_datasets,create_non_IID_datasets_strict
from CFL.model import SimpleCNN,EMNISTNet,CIFAR10Net
from CFL.fedavg import Server, Client

class LatticeQuantizer:
    def __init__(self, bits=4, lattice_dim=2, use_dither=False, batch_size=128):
        self.bits = bits
        self.lattice_dim = lattice_dim
        self.use_dither = use_dither
        self.batch_size = batch_size
        self.codebook = self._build_codebook()
        self._seed = None

    def set_dither_seed(self, seed: int):
        self._seed = seed

    def _build_codebook(self):
        qmax = 2**(self.bits - 1) - 1
        if self.lattice_dim == 1:
            codebook = torch.arange(-qmax, qmax + 1, dtype=torch.float32).unsqueeze(1)
        elif self.lattice_dim == 2:
            points = []
            for i in range(-qmax, qmax + 1):
                for j in range(-qmax, qmax + 1):
                    points.append([i, j])
            codebook = torch.tensor(points, dtype=torch.float32)
        else:
            raise NotImplementedError("Only 1D and 2D lattices supported.")
        return codebook

    def _find_nearest_lattice_point(self, vectors):
        device = vectors.device
        codebook_cpu = self.codebook.cpu()
        vectors_cpu = vectors.detach().cpu()
        num_vectors = vectors.shape[0]
        nearest_points = []
        for start in range(0, num_vectors, self.batch_size):
            end = min(start + self.batch_size, num_vectors)
            batch = vectors_cpu[start:end].unsqueeze(1)
            dist = torch.norm(batch - codebook_cpu.unsqueeze(0), dim=2)
            idx = dist.argmin(dim=1)
            nearest_points.append(codebook_cpu[idx])
        result_cpu = torch.cat(nearest_points, dim=0)
        return result_cpu.to(device)

    def quantize_dequantize(self, vectors):
        if self.use_dither:
            rng_state = torch.get_rng_state()
            if self._seed is not None:
                torch.manual_seed(self._seed)
            dither = torch.rand_like(vectors) - 0.5
            vectors = vectors + dither
            torch.set_rng_state(rng_state)
        quantized = self._find_nearest_lattice_point(vectors)
        return quantized

class CFLClient(Client):
    def __init__(self, client_id, train_loader, device, lr=0.001,
                 quantizer_bits=8, use_dither=False, num_classes=10):
        super().__init__(client_id, train_loader, device, lr=lr)
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.lr = lr
        self.model = None
        self.optimizer = None
        self.quantizer = LatticeQuantizer(bits=quantizer_bits, use_dither=use_dither)
        self.gradient_norm = 0.0
        self.num_classes = num_classes
        self.local_training_time = 0.0
        self.local_flops = 0
        self.single_iter_flops = 0
        self.last_train_flops = 0
        self.last_quant_flops = 0

    def receive_model(self, global_model, dither_seed=None):
        if self.model is None:
            try:
                self.model = copy.deepcopy(global_model).to(self.device)
            except Exception:
                try:
                    self.model = SimpleCNN(num_classes=self.num_classes).to(self.device)
                    self.model.load_state_dict(global_model.state_dict())
                except Exception:
                    raise RuntimeError("无法初始化 client.model，请检查 global_model 或 SimpleCNN")
        else:
            self.model.load_state_dict(global_model.state_dict())
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.initial_params = {n: p.detach().clone().cpu()
                               for n, p in self.model.named_parameters() if p.requires_grad}
        if dither_seed is not None:
            self.quantizer.set_dither_seed(dither_seed)
        try:
            model_cpu = copy.deepcopy(self.model).cpu()
            try:
                sample = next(iter(self.train_loader))[0]
                input_size = tuple(sample.shape[1:])
                batch_size = sample.shape[0]
            except Exception:
                input_size = (1, 28, 28)
                batch_size = 1
            self.single_iter_flops = estimate_flops(model_cpu, input_size=input_size,
                                                    batch_size=batch_size, device='cpu')
        except Exception:
            self.single_iter_flops = 0

    def compute_gradient_norm(self, max_batches=1):
        if self.model is None:
            return 0.0
        self.model.train()
        total_sq = 0.0
        seen = 0
        it = iter(self.train_loader)
        for _ in range(max_batches):
            try:
                data, target = next(it)
            except StopIteration:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad(set_to_none=True)
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            sq = sum(p.grad.detach().float().pow(2).sum().item()
                     for p in self.model.parameters() if p.grad is not None)
            total_sq += sq
            seen += 1
            for p in self.model.parameters():
                p.grad = None
        self.gradient_norm = math.sqrt(total_sq / max(1, seen))
        return self.gradient_norm

    def local_train(self, epochs=1, clip_grad=10.0):
        if self.model is None:
            raise RuntimeError("Client model not initialized. Call receive_model first.")
        self.model.train()
        start_time = time.time()
        flops_this_round = 0
        if not hasattr(self, 'initial_params') or self.initial_params is None:
            self.initial_params = {n: p.detach().clone().cpu()
                                   for n, p in self.model.named_parameters() if p.requires_grad}
        total_iterations = 0
        for e in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = F.cross_entropy(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                self.optimizer.step()
                total_iterations += 1
        training_time = time.time() - start_time
        self.local_training_time += training_time
        flops_this_round = int(self.single_iter_flops * total_iterations)
        self.local_flops += flops_this_round
        update = OrderedDict()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                cur = p.detach().cpu()
                init = self.initial_params.get(n, torch.zeros_like(cur))
                update[n] = (cur - init).to(self.device)
        quantized_update = OrderedDict()
        quant_flops = 0
        for n, upd in update.items():
            v = upd.view(-1, self.quantizer.lattice_dim)
            qv = self.quantizer.quantize_dequantize(v)
            quantized_update[n] = qv.view_as(upd)
            n_elem = v.numel()
            quant_flops += n_elem * (2 * self.quantizer.lattice_dim)
        total_flops = flops_this_round + quant_flops
        self.local_flops += total_flops
        self.last_train_flops = flops_this_round
        self.last_quant_flops = quant_flops
        try:
            num_samples = len(self.train_loader.dataset)
        except Exception:
            num_samples = sum(b[0].shape[0] for b in self.train_loader)
        return quantized_update, training_time, flops_this_round,quant_flops, num_samples

class CFLServer(Server):
    def __init__(self, test_loader, device, num_classes=10, global_model=None,
                 resource_blocks=10, bandwidth=2e6):
        if isinstance(Server, type) and Server is not object:
            try:
                super().__init__(test_loader, device, num_classes)
            except Exception:
                self.test_loader = test_loader
                self.device = device
                self.global_model = SimpleCNN(num_classes=num_classes).to(device)
                self.clients = []
        else:
            self.test_loader = test_loader
            self.device = device
            self.global_model = SimpleCNN(num_classes=num_classes).to(device)
            self.clients = []
        self.resource_blocks = resource_blocks
        self.bandwidth = bandwidth
        self.history = {
            'accuracy': [],
            'loss': [],
            'communication_rounds': 0,
            'communication_cost': [],
            'computation_cost': [],
            'per_round_time': [],
            'total_comm_bytes': 0,
            'total_computation_flops': 0,
            'total_training_time': 0.0
        }
        self.model_size_bytes = sum(p.numel() * p.element_size()
                                    for p in self.global_model.parameters() if p.requires_grad)

    def attach_clients(self, clients):
        self.clients = clients

    def probabilistic_device_selection(self):
        grad_norms = [c.gradient_norm for c in self.clients]
        total = sum(grad_norms)
        if total > 0:
            probs_grad = [g / total for g in grad_norms]
        else:
            probs_grad = [1.0 / max(1, len(self.clients))] * len(self.clients)
        k = min(self.resource_blocks, len(self.clients))
        chosen_idx = np.random.choice(len(self.clients), size=k, replace=False, p=probs_grad)
        selected = [self.clients[i] for i in chosen_idx]
        return selected, probs_grad

    def wireless_resource_allocation(self, selected_clients):
        allocation = {c.client_id: i % self.resource_blocks for i, c in enumerate(selected_clients)}
        return allocation

    def aggregate(self, client_updates):
        if len(client_updates) == 0:
            return
        updates, ns = zip(*client_updates)
        total = sum(ns)
        avg = OrderedDict()
        for key in updates[0].keys():
            avg[key] = torch.zeros_like(updates[0][key], device=self.device)
        for upd, n in zip(updates, ns):
            w = n / total if total > 0 else 1.0 / len(updates)
            for k in avg.keys():
                avg[k] += upd[k].to(self.device) * w
        with torch.no_grad():
            for name, p in self.global_model.named_parameters():
                if name in avg:
                    p.add_(avg[name].to(p.device).to(p.dtype))

    def evaluate(self):
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.global_model(data)
                test_loss += F.cross_entropy(out, target, reduction='sum').item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        acc = 100.0 * correct / len(self.test_loader.dataset)
        self.history['loss'].append(test_loss)
        self.history['accuracy'].append(acc)
        self.history['communication_rounds'] += 1
        print(f"轮次 {self.history['communication_rounds']}: 损失: {test_loss:.4f}, 准确率: {acc:.2f}%")
        return test_loss, acc

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.4):
        print("开始训练 (CSFL-style with lattice quant) ...")
        start_all = time.time()
        for r in range(communication_rounds):
            round_start = time.time()
            num_selected = max(1, int(len(self.clients) * client_frac))
            dither_seed = int(time.time() * 1000) % (2 ** 32 - 1)
            for c in self.clients:
                c.receive_model(self.global_model, dither_seed)
                try:
                    c.compute_gradient_norm(max_batches=1)
                except Exception:
                    pass
            selected, probs = self.probabilistic_device_selection()
            allocation = self.wireless_resource_allocation(selected)
            print(f"Round {r + 1}, selected clients: {[c.client_id for c in selected]}")
            client_updates = []
            round_comp_time = 0.0
            round_comp_flops = 0.0
            round_flops_model = 0
            round_flops_qua = 0
            uplink_bytes = 0
            downlink_bytes = 0
            for c in selected:
                upd, ttime, model_flops,quant_flops, ns = c.local_train(epochs=local_epochs)
                client_updates.append((upd, ns))
                round_comp_time += ttime
                round_flops_model += model_flops
                round_flops_qua += quant_flops
                num_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
                bits = max(1, c.quantizer.bits)
                uplink_bytes += int(num_params * bits / 8.0)
                downlink_bytes += self.model_size_bytes
            round_comp_flops = round_flops_model+round_flops_qua
            round_comm = uplink_bytes + downlink_bytes
            self.history['communication_cost'].append(round_comm)
            self.history['computation_cost'].append(round_comp_flops)
            self.history['per_round_time'].append(round_comp_time)
            self.history['total_comm_bytes'] += round_comm
            self.history['total_computation_flops'] += round_comp_flops
            self.aggregate(client_updates)
            self.evaluate()
            print(
                f"[Round {r + 1}] 通信: {round_comm / (1024 ** 2):.4f} MB, "
                f"模型开销: {round_flops_model / 1e9:.4f} GFLOPs, "
                f"量化开销: {round_flops_qua} FLOPs, "
                f"计算: {round_comp_flops / 1e9:.4f} GFLOPs, "
                f"时间: {time.time() - round_start:.2f}s"
            )
        self.history['total_training_time'] = time.time() - start_all
        print("训练结束")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "CFL.txt")

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
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--communication_rounds', type=int, default=3)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--quantizer_bits', type=int, default=8)
    parser.add_argument('--use_dither', action='store_true',default=True)
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--classes_per_client', type=int, default=3)
    parser.add_argument('--resource_blocks', type=int, default=10)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
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
        loader = torch.utils.data.DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        c = CFLClient(i, loader, device, lr=args.lr, quantizer_bits=args.quantizer_bits, use_dither=args.use_dither,
                      num_classes=10)
        clients.append(c)
    num_classes = 10
    global_model = SimpleCNN(num_classes=num_classes).to(device)
    server = CFLServer(test_loader, device, global_model, resource_blocks=args.resource_blocks)
    server.attach_clients(clients)
    print("Initial evaluation:")
    server.evaluate()
    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs, client_frac=0.4)

if __name__ == "__main__":
    main()
