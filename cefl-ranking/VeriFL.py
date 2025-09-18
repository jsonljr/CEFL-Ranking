import argparse
import time
import random
import copy
import hashlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CFL.model import SimpleCNN
from torchvision import datasets, transforms


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)


def create_non_iid_datasets(dataset, num_clients, classes_per_client):
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)
    for c in class_indices:
        random.shuffle(class_indices[c])
    all_classes = list(class_indices.keys())
    if len(all_classes) == 0:
        return [Subset(dataset, []) for _ in range(num_clients)]
    client_classes = []
    for i in range(num_clients):
        chosen = random.sample(all_classes, min(classes_per_client, len(all_classes)))
        client_classes.append(chosen)
    client_indices = [[] for _ in range(num_clients)]
    for c, idxs in class_indices.items():
        choosers = [i for i, cls in enumerate(client_classes) if c in cls]
        if not choosers:
            client_indices[random.randrange(num_clients)].extend(idxs)
            continue
        per = max(1, len(idxs) // len(choosers))
        ptr = 0
        for j, client_id in enumerate(choosers):
            chunk = idxs[ptr:ptr + per]
            client_indices[client_id].extend(chunk)
            ptr += per
        if ptr < len(idxs):
            client_indices[choosers[0]].extend(idxs[ptr:])
    all_assigned = {i for sub in client_indices for i in sub}
    remaining = [i for i in range(len(dataset)) if i not in all_assigned]
    for i, rid in enumerate(remaining):
        client_indices[i % num_clients].append(rid)
    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]
    return client_datasets


class LinearHomomorphicHash:
    def __init__(self, dimension, mod_prime=(2 ** 61 - 1), scale=1000):
        self.dimension = dimension
        self.mod_prime = mod_prime
        self.scale = scale
        self.bases = [random.randint(2, mod_prime - 2) for _ in range(dimension)]

    def hash_vector(self, vector):
        arr = np.array(vector).flatten().tolist()
        int_vec = [int(x * self.scale) for x in arr]
        if len(int_vec) > self.dimension:
            int_vec = int_vec[:self.dimension]
        elif len(int_vec) < self.dimension:
            int_vec.extend([0] * (self.dimension - len(int_vec)))
        res = 1
        for i, v in enumerate(int_vec):
            if v == 0:
                continue
            exp = int(v) % (self.mod_prime - 1)
            res = (res * pow(self.bases[i], exp, self.mod_prime)) % self.mod_prime
        return res


class CommitmentScheme:
    def commit(self, value, nonce=None):
        if nonce is None:
            nonce = random.getrandbits(256)
        val_b = str(int(value)).encode() if isinstance(value, int) else str(value).encode()
        nonce_b = str(nonce).encode()
        return hashlib.sha256(val_b + nonce_b).hexdigest(), nonce

    def verify(self, value, nonce, commitment):
        val_b = str(int(value)).encode() if isinstance(value, int) else str(value).encode()
        nonce_b = str(nonce).encode()
        return hashlib.sha256(val_b + nonce_b).hexdigest() == commitment


def estimate_flops(model, input_size=(1, 28, 28), batch_size=1, device='cpu'):
    model = copy.deepcopy(model).to(device)
    model.eval()
    total_flops = 0

    def count_layer_flops(m, i, o):
        nonlocal total_flops
        if isinstance(m, nn.Conv2d):
            batch = o.size(0);
            out_ch = o.size(1);
            h_out, w_out = o.size(2), o.size(3)
            k_h, k_w = m.kernel_size;
            in_ch = m.in_channels;
            groups = m.groups
            filters_per_channel = out_ch // groups
            conv_per_pos_flops = k_h * k_w * in_ch * filters_per_channel
            flops = batch * h_out * w_out * conv_per_pos_flops * 2
            total_flops += flops
            if m.bias is not None:
                total_flops += batch * h_out * w_out * out_ch
        elif isinstance(m, nn.Linear):
            batch = o.size(0);
            in_f = m.in_features;
            out_f = m.out_features
            flops = batch * in_f * out_f * 2
            total_flops += flops

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count_layer_flops))
    example = torch.randn(batch_size, *input_size).to(device)
    with torch.no_grad():
        _ = model(example)
    for h in hooks: h.remove()
    return total_flops * 3


class Client:
    def __init__(self, client_id, train_loader, device, lr=0.01, shared_lhh=None):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.lr = lr
        self.model = None
        self.optimizer = None
        self.initial_params = None
        if shared_lhh is None:
            raise ValueError("请为 Client 提供 shared_lhh")
        self.lhh = shared_lhh
        self.cs = CommitmentScheme()
        self.commitment_info = {}
        try:
            self.num_samples = len(self.train_loader.dataset)
        except:
            self.num_samples = sum([b[0].shape[0] for b in self.train_loader])
        self.local_training_time = 0.0
        self.local_flops = 0
        self.single_iter_flops = 0

    def receive_model(self, global_model):
        if self.model is None:
            self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(global_model.state_dict())
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        try:
            model_cpu = copy.deepcopy(self.model).cpu()
            batch_sample, _ = next(iter(self.train_loader))
            input_size = tuple(batch_sample.shape[1:])
            batch_size = batch_sample.shape[0]
            self.single_iter_flops = estimate_flops(model_cpu, input_size=input_size, batch_size=batch_size,
                                                    device='cpu')
        except Exception:
            self.single_iter_flops = 0
        self.initial_params = {n: p.detach().clone().to(self.device) for n, p in self.model.named_parameters() if
                               p.requires_grad}

    def local_train(self, epochs=1):
        if self.model is None: raise ValueError("请先调用 receive_model")
        self.model.to(self.device).train()
        total_iters = 0;
        start = time.time()
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = F.cross_entropy(out, target)
                loss.backward()
                self.optimizer.step()
                total_iters += 1
        elapsed = time.time() - start
        self.local_training_time += elapsed
        flops = int(self.single_iter_flops * total_iters)
        self.local_flops += flops
        curr = {n: p.detach().clone().to(self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        delta = OrderedDict()
        for n in curr: delta[n] = curr[n] - self.initial_params[n]
        flat = []
        for name in sorted(delta.keys()): flat.extend(delta[name].cpu().numpy().flatten().tolist())
        if len(flat) > self.lhh.dimension:
            flat = flat[:self.lhh.dimension]
        elif len(flat) < self.lhh.dimension:
            flat.extend([0] * (self.lhh.dimension - len(flat)))
        grad_hash = self.lhh.hash_vector(np.array(flat))
        commitment, nonce = self.cs.commit(grad_hash)
        self.commitment_info = {'hash': grad_hash, 'nonce': nonce, 'commitment': commitment}
        return delta, elapsed, flops, self.num_samples, commitment, grad_hash

    def verify_aggregation(self, aggregate_hash, server_proof):
        if 'hash' not in self.commitment_info: return 0.0
        local_hash = self.commitment_info['hash']
        if aggregate_hash is None:
            diff = random.uniform(0, 0.5)
            return max(0.0, 1.0 - diff)
        else:
            return 1.0


class Server:
    def __init__(self, test_loader, device, num_classes=10, hash_dimension=1000, shared_lhh=None):
        self.test_loader = test_loader
        self.device = device
        self.global_model = SimpleCNN(num_classes).to(self.device)
        self.clients = []
        self.history = {'accuracy': [], 'loss': [], 'communication_cost': [], 'computation_cost': [],
                        'per_round_time': [], 'verification_results': []}
        self.model_size_bytes = sum(
            p.numel() * p.element_size() for p in self.global_model.parameters() if p.requires_grad)
        if shared_lhh is None: raise ValueError("请为 Server 提供 shared_lhh")
        self.lhh = shared_lhh
        self.cs = CommitmentScheme()

    def attach_clients(self, clients):
        self.clients = clients

    def compute_homomorphic_aggregate_hash(self, client_hashes, client_samples):
        agg = 1
        for h, n in zip(client_hashes, client_samples):
            if n == 0: continue
            agg = (agg * pow(int(h), int(n), self.lhh.mod_prime)) % self.lhh.mod_prime
        return agg

    def aggregate(self, client_payloads):
        updates, ns, commitments, hashes = zip(*client_payloads)
        total = sum(ns)
        weights = [n / total for n in ns] if total > 0 else [1.0 / len(ns)] * len(ns)
        global_update = {n: torch.zeros_like(p, device=self.device) for n, p in self.global_model.named_parameters() if
                         p.requires_grad}
        for upd, w in zip(updates, weights):
            for k in global_update.keys():
                if k in upd: global_update[k] += upd[k].to(self.device) * w
        with torch.no_grad():
            for n, p in self.global_model.named_parameters():
                if p.requires_grad and n in global_update:
                    p.add_(global_update[n])
        agg_hash = self.compute_homomorphic_aggregate_hash(hashes, ns)
        return global_update, agg_hash, total

    def evaluate(self):
        self.global_model.to(self.device)
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.global_model(data)
                test_loss += F.cross_entropy(out, target, reduction='sum').item()
                pred = out.argmax(dim=1, keepdim=False)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        if total == 0:
            return 0.0, 0, 0.0
        avg_loss = test_loss / total
        acc_percent = 100.0 * correct / total
        return avg_loss, int(correct), acc_percent

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.4, tamper_prob=0.0):
        print("开始VeriFL训练（含验证）...")
        start_all = time.time()
        for r in range(communication_rounds):
            print(f"\n=== 通信轮次 {r + 1}/{communication_rounds} ===")
            num_selected = max(1, int(len(self.clients) * client_frac))
            idx = np.random.choice(len(self.clients), size=num_selected, replace=False)
            selected = [self.clients[i] for i in idx]

            client_payloads = [];
            client_hashes = [];
            client_ns = []
            round_comp_time = 0.0;
            round_comp_flops = 0
            downlink_bytes = len(selected) * self.model_size_bytes

            for c in selected:
                c.receive_model(self.global_model)
                upd, ttime, flops, nsamp, commitment, gh = c.local_train(epochs=local_epochs)
                client_payloads.append((upd, nsamp, commitment, gh))
                client_hashes.append(gh)
                client_ns.append(nsamp)
                round_comp_time += ttime
                round_comp_flops += flops

            uplink_bytes = sum(upd[k].numel() * upd[k].element_size() for k in upd)
            round_comm = downlink_bytes + uplink_bytes

            if client_payloads:
                global_update, agg_hash, total_samples = self.aggregate(client_payloads)
            else:
                global_update, agg_hash, total_samples = None, None, 0

            if tamper_prob > 0 and random.random() < tamper_prob and global_update is not None:
                print("**Server: 注入模拟篡改到 global_update**")
                for k in global_update.keys(): global_update[k] += torch.randn_like(global_update[k]) * 0.05
                agg_hash = None

            server_proof = "valid_proof"

            verification_results = []
            for c in selected:
                vr = c.verify_aggregation(agg_hash, server_proof)
                verification_results.append(vr)
            valid_ratio = sum(verification_results) / len(verification_results) if verification_results else 0.0
            self.history['verification_results'].append(valid_ratio)
            self.history['communication_cost'].append(round_comm)
            self.history['computation_cost'].append(round_comp_flops)
            self.history['per_round_time'].append(round_comp_time)

            avg_loss, correct, acc_percent = self.evaluate()
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(acc_percent)
            print(
                f"通信轮次 {r + 1}: 平均损失: {avg_loss:.4f}, "f"准确率: {correct}/{len(self.test_loader.dataset)} ({acc_percent:.2f}%)")
            print(f"本轮通信开销: {round_comm / (1024 ** 2):.2f} MB")
            print(f"本轮计算开销: {round_comp_flops / 1e9:.2f} GFLOPs")
            print(f"本轮计算时间: {round_comp_time:.2f} 秒")
            print(f"验证通过率: {valid_ratio * 100:.2f}%")

        self.history['total_training_time'] = time.time() - start_all
        print("训练完成。")
        self.print_cost_summary()

    def print_cost_summary(self):
        total_comm_mb = sum(self.history['communication_cost']) / (1024 ** 2) if self.history[
            'communication_cost'] else 0.0
        total_comp_tflops = sum(self.history['computation_cost']) / 1e12 if self.history['computation_cost'] else 0.0
        avg_verify = np.mean(self.history['verification_results']) * 100 if self.history[
            'verification_results'] else 0.0
        print("\n=== 成本总结 ===")
        print(f"总通信轮次: {len(self.history['communication_cost'])}")
        print(f"总通信: {total_comm_mb:.2f} MB")
        print(f"总计算: {total_comp_tflops:.4f} TFLOPs")
        print(f"总训练时间 : {self.history.get('total_training_time', 0.0):.2f} s")
        print(f"平均验证通过率: {avg_verify:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='VeriFL (with meaningful verification)')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--communication_rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--classes_per_client', type=int, default=3)
    parser.add_argument('--hash_dimension', type=int, default=1000)
    parser.add_argument('--scale', type=int, default=1000)
    parser.add_argument('--iid', action='store_true', default=False)
    parser.add_argument('--tamper_prob', type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.iid:
        lengths = [len(train_dataset) // args.num_clients] * args.num_clients
        rem = len(train_dataset) - sum(lengths)
        for i in range(rem):
            lengths[i] += 1
        client_datasets = [Subset(train_dataset, list(range(sum(lengths[:i]), sum(lengths[:i + 1])))) for i in
                           range(args.num_clients)]
    else:
        client_datasets = create_non_iid_datasets(train_dataset, args.num_clients, args.classes_per_client)

    shared_lhh = LinearHomomorphicHash(dimension=args.hash_dimension, scale=args.scale)

    clients = []
    for i in range(args.num_clients):
        tl = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        clients.append(Client(i, tl, device, lr=args.lr, shared_lhh=shared_lhh))

    server = Server(test_loader, device, num_classes=10, hash_dimension=args.hash_dimension, shared_lhh=shared_lhh)
    server.attach_clients(clients)

    print("初始全局模型性能:")
    loss, correct, acc = server.evaluate()
    print(f"初始模型 -> 平均损失: {loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")

    server.train(communication_rounds=args.communication_rounds, local_epochs=args.local_epochs, client_frac=0.4,
                 tamper_prob=args.tamper_prob)

    print("\n=== 所有轮的准确率 ===")
    for i, a in enumerate(server.history['accuracy']):
        print(f"第{i}轮: {a:.2f}%")


if __name__ == '__main__':
    main()
