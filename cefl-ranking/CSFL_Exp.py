import argparse
import time
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from CFL.model import SimpleCNN
from CFL.fedavg import Client, Server
from CFL.dataset import create_non_iid_datasets

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

def model_to_vector(model):
    vecs, shapes, names = [], [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            vecs.append(p.data.view(-1).to(torch.float32))
            shapes.append(p.data.shape)
            names.append(name)
    full = torch.cat(vecs) if vecs else torch.tensor([], dtype=torch.float32)
    return full, shapes, names

def vector_to_named_update(vec, shapes, names, device):
    out = OrderedDict()
    idx = 0
    for s, n in zip(shapes, names):
        num = int(np.prod(s))
        chunk = vec[idx:idx+num].view(s).to(device)
        out[n] = chunk
        idx += num
    return out

def add_update_to_model(global_model, update_named):
    with torch.no_grad():
        for name, p in global_model.named_parameters():
            if name in update_named and p.requires_grad:
                p.add_(update_named[name].to(p.device))

def compressed_sensing_Ax(x, m, seed=None, device='cpu', block_size=200):
    n = x.numel()
    y = torch.zeros(m, device=device)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed) % (2**32 - 1))
    for i in range(0, m, block_size):
        sz = min(block_size, m - i)
        for j in range(0, n, block_size):
            col_sz = min(block_size, n - j)
            A_sub = torch.randn(sz, col_sz, generator=g, device=device) / np.sqrt(m)
            y[i:i+sz] += A_sub @ x[j:j+col_sz]
    return y

def compressed_sensing_ATz(z, m, n, seed=None, device='cpu', block_size=200):
    x_grad = torch.zeros(n, device=device)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed) % (2**32 - 1))
    for i in range(0, m, block_size):
        sz = min(block_size, m - i)
        for j in range(0, n, block_size):
            col_sz = min(block_size, n - j)
            A_sub = torch.randn(sz, col_sz, generator=g, device=device) / np.sqrt(m)
            z_block = z[i:i+sz]
            x_grad[j:j+col_sz] += A_sub.t() @ z_block
    return x_grad

def iterative_hard_thresholding_block(y, n, k, m, seed=None, iters=20, x0=None, device='cuda', block_size=200):
    if x0 is None:
        x = torch.zeros(n, device=device)
    else:
        x = x0.clone().to(device)
    y = y.to(device)
    for _ in range(iters):
        res = y - compressed_sensing_Ax(x, m, seed=seed, device=device, block_size=block_size)
        grad = compressed_sensing_ATz(res, m, n, seed=seed, device=device, block_size=block_size)
        x = x + grad
        if k > 0:
            flat = x.view(-1)
            if k < flat.numel():
                topk_vals, topk_idx = torch.topk(flat.abs(), k)
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask[topk_idx] = True
                new_flat = torch.zeros_like(flat)
                new_flat[mask] = flat[mask]
                x = new_flat.view_as(x)
    return x

class CSFLCompressClient(Client):
    def __init__(self, client_id, train_loader, device, lr=0.01,
                 sparsity_frac=0.01, measure_rate=0.01, resid_frac=0.01, iht_iters=20):
        super().__init__(client_id, train_loader, device, lr)
        self.sparsity_frac = sparsity_frac
        self.measure_rate = measure_rate
        self.resid_frac = resid_frac
        self.iht_iters = iht_iters

    def local_train_stage1(self, global_model):
        delta, training_time, flops, num_samples = super().local_train(epochs=1)
        local_vec, shapes, names = model_to_vector(self.model)
        global_vec, _, _ = model_to_vector(global_model)
        device = self.device
        delta_vec = (local_vec - global_vec).to(device)
        n = delta_vec.numel()
        k = max(1, int(self.sparsity_frac * n))

        flat = delta_vec.view(-1)
        if k < flat.numel():
            topk_vals, topk_idx = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[topk_idx] = True
            sparse_flat = torch.zeros_like(flat)
            sparse_flat[mask] = flat[mask]
        else:
            sparse_flat = flat.clone()
            topk_idx = torch.arange(flat.numel(), dtype=torch.long, device=device)
        x_sparse = sparse_flat.view(-1).to(device)
        seed = int(time.time() * 1e6) ^ (self.client_id + 12345)
        m = max(1, int(self.measure_rate * n))
        y = compressed_sensing_Ax(x_sparse, m, seed=seed, device=device)

        payload_localstore = {
            'x_sparse': x_sparse,
            'indices': topk_idx,
            'shapes': shapes,
            'names': names
        }

        payload_stage1 = {
            'y': y,
            'seed': seed,
            'n': n,
            'k': k,
            'm': m,
            'num_samples': num_samples
        }
        self._local_stage1_store = payload_localstore
        return payload_stage1, training_time, flops, num_samples

    def local_stage2_send_sign(self, global_model, x_hat_vec):
        x_hat = x_hat_vec.to(self.device)
        x_sparse = self._local_stage1_store['x_sparse']
        resid = (x_sparse - x_hat).view(-1)
        n = resid.numel()
        k_res = max(1, int(self.resid_frac * n))
        flat = resid.view(-1)
        if k_res < n:
            top_vals, top_idx = torch.topk(flat.abs(), k_res)
        else:
            top_idx = torch.arange(n, dtype=torch.long, device=self.device)
        signs = torch.sign(flat[top_idx]).cpu().to(torch.int8)
        mean_abs = flat[top_idx].abs().mean().item() if top_idx.numel() > 0 else 0.0
        payload_stage2 = {
            'sign_indices': top_idx.cpu(),
            'signs': signs,
            'mean_abs': mean_abs,
            'num_samples': 0
        }
        return payload_stage2

class CSFLCompressServer(Server):
    def __init__(self, clients, device, test_loader=None, global_model=None, num_classes=10):
        super().__init__(device=device, test_loader=test_loader, global_model=global_model, num_classes=num_classes)
        self.clients = clients

    def train(self, communication_rounds=10, local_epochs=1, client_frac=0.4, iht_iters=20):
        start_time = time.time()
        for r in range(communication_rounds):
            round_start_time = time.time()
            num_selected = max(1, int(len(self.clients) * client_frac))
            idx = np.random.choice(len(self.clients), size=num_selected, replace=False)
            selected_clients = [self.clients[i] for i in idx]

            downlink_bytes = 0
            for client in selected_clients:
                client.receive_model(self.global_model)
                for p in self.global_model.parameters():
                    downlink_bytes += p.numel() * p.element_size()

            client_stage1_payloads = []
            uplink_bytes = 0
            round_comp_flops = 0
            for client in selected_clients:
                payload1, t_time, flops, num_samples = client.local_train_stage1(self.global_model)
                client_stage1_payloads.append((payload1, num_samples, client))
                uplink_bytes += payload1['y'].numel() * payload1['y'].element_size()
                round_comp_flops += flops

            ys = []
            wts = []
            total_samples = sum([p[1] for p in client_stage1_payloads])
            for (payload1, ns, client) in client_stage1_payloads:
                ys.append(payload1['y'].float())
                wts.append(ns / total_samples if total_samples > 0 else 1.0 / len(client_stage1_payloads))
            ys_stack = torch.stack(ys, dim=0)
            wts_tensor = torch.tensor(wts, dtype=torch.float32).view(-1, 1)
            y_bar = (wts_tensor.to(ys_stack.device) * ys_stack).sum(dim=0)

            meta = client_stage1_payloads[0][0]
            seed, n, k, m = meta['seed'], meta['n'], meta['k'], meta['m']
            y_bar = y_bar.to(self.device)
            x_hat = iterative_hard_thresholding_block(
                y=y_bar,
                n=n,
                k=k,
                m=m,
                seed=seed,
                iters=iht_iters,
                device=self.device
            )

            for (payload1, ns, client) in client_stage1_payloads:
                payload2 = client.local_stage2_send_sign(self.global_model, x_hat)
                payload2['num_samples'] = ns
                uplink_bytes += payload2['sign_indices'].numel() * payload2['sign_indices'].element_size()
                uplink_bytes += payload2['signs'].numel() * payload2['signs'].element_size()

            final_update_vec = x_hat
            shapes = client_stage1_payloads[0][2]._local_stage1_store['shapes']
            names = client_stage1_payloads[0][2]._local_stage1_store['names']
            update_named = vector_to_named_update(final_update_vec.cpu(), shapes, names, self.device)
            add_update_to_model(self.global_model, update_named)

            round_comp_time = time.time() - round_start_time
            round_comm_bytes = downlink_bytes + uplink_bytes
            self.history['communication_cost'].append(round_comm_bytes)
            self.history['computation_cost'].append(round_comp_flops)
            self.history['per_round_time'].append(round_comp_time)
            self.history['total_comm_bytes'] += round_comm_bytes
            self.history['total_computation_flops'] += round_comp_flops

            self.evaluate()
            print(f"[Round {r + 1}] 通信: {round_comm_bytes / 1024 ** 2:.4f} MB, "
                  f"计算: {round_comp_flops / 1e9:.4f} GFLOPs, "
                  f"时间: {round_comp_time:.2f}s")

        self.history['total_training_time'] = time.time() - start_time
        print("训练完成!")
        self.print_cost_summary()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--communication_rounds', type=int, default=3)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iid', action='store_true', default=False)
    parser.add_argument('--classes_per_client', type=int, default=3)
    parser.add_argument('--measure_rate', type=float, default=0.01)
    parser.add_argument('--sparsity_frac', type=float, default=0.01)
    parser.add_argument('--resid_frac', type=float, default=0.01)
    parser.add_argument('--iht_iters', type=int, default=20)
    parser.add_argument('--client_frac', type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.iid:
        from torch.utils.data import random_split
        client_datasets = random_split(
            train_dataset,
            [len(train_dataset) // args.num_clients] * args.num_clients
        )
    else:
        client_datasets = create_non_iid_datasets(
            train_dataset,
            args.num_clients,
            args.classes_per_client
        )

    clients = []
    for i in range(args.num_clients):
        loader = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        client = CSFLCompressClient(
            client_id=i,
            train_loader=loader,
            device=device,
            lr=args.lr,
            sparsity_frac=args.sparsity_frac,
            measure_rate=args.measure_rate,
            resid_frac=args.resid_frac,
            iht_iters=args.iht_iters
        )
        clients.append(client)

    num_classes = 10
    global_model = SimpleCNN(num_classes=num_classes).to(device)
    server = CSFLCompressServer(
        clients=clients,
        device=device,
        test_loader=test_loader,
        global_model=global_model,
        num_classes=num_classes
    )

    server.test_loader = test_loader

    print("初始全局模型性能:")
    server.evaluate()

    server.train(
        communication_rounds=args.communication_rounds,
        local_epochs=args.local_epochs,
        client_frac=args.client_frac,
        iht_iters=args.iht_iters
    )

    print("\n=== 每轮准确率 ===")
    for i, acc in enumerate(server.history['accuracy']):
        print(f"第{i}轮: {acc:.2f}%")

if __name__ == '__main__':
    main()
