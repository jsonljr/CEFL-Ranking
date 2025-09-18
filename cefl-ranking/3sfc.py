import argparse
import random
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
from torchvision import datasets, transforms
from CFL.dataset import create_non_iid_datasets
from CFL.model import SimpleCNN
from CFL.fedavg import estimate_flops
import time


def train_op(model, loader, optimizer, epochs=1, loss_fn=torch.nn.CrossEntropyLoss(), device="cpu"):
    model.train()
    for _ in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            loss.backward()
            optimizer.step()
    return running_loss / samples

def eval_op(model, loader, device="cpu"):
    model.eval()
    samples, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)
            samples += y.shape[0]
            correct += (predicted == y).sum().item()
    return correct / samples

def copy_state(target, source):
    for name in target:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

def add_(target, added, addend):
    for name in target:
        target[name].data = added[name].data.clone() + addend[name].data.clone()

def reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp

class Client:
    def __init__(self, client_id, train_loader, device, lr=0.01, model_fn=None, n_classes=10):
        self.id = client_id
        self.device = device
        self.model = model_fn().to(device)
        self.train_loader = train_loader
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.0)
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.W_old = {k: v.clone() for k, v in self.W.items()}
        self.dW = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.dW_residual = {k: torch.zeros_like(v) for k, v in self.W.items()}

    def synchronize_with_server(self, server):
        copy_state(self.W, server.W)

    def compute_weight_update(self, epochs=1):
        copy_state(self.W_old, self.W)
        loss = train_op(self.model, self.train_loader, self.optimizer, epochs=epochs, loss_fn=self.loss_fn, device=self.device)
        subtract_(self.dW, self.W, self.W_old)
        add_(self.dW_residual, self.dW_residual, self.dW)
        return loss

    def reset(self):
        copy_state(self.W, self.W_old)

    def compute_fedsynth(self, n_sample, n_classes, eta_w, eta, epochs=1):
        synthetic_input_size = [20, n_sample] + list(next(iter(self.train_loader))[0].shape[1:])
        synthetic_inputs = torch.randn(tuple(synthetic_input_size), device=self.device, requires_grad=True)
        synthetic_labels = torch.randn((20, n_sample, n_classes), device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([synthetic_inputs, synthetic_labels], lr=eta, momentum=0.0)

        for _ in range(epochs):
            synthetic_model = deepcopy(self.model)
            synthetic_model.train()
            synthetic_optim = torch.optim.SGD(synthetic_model.parameters(), lr=eta_w, momentum=0.0)
            for i in range(20):
                synthetic_optim.zero_grad()
                inputs_batch = synthetic_inputs[i].view(-1, *synthetic_inputs.shape[2:])  # [n_sample, 1, 28,28]
                labels_batch = torch.max(synthetic_labels[i], 1)[1]  # [n_sample]
                loss = torch.nn.CrossEntropyLoss()(synthetic_model(inputs_batch), labels_batch)
                loss.backward()
                synthetic_optim.step()

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                for i in range(20):
                    inputs_batch = synthetic_inputs[i].view(-1, *synthetic_inputs.shape[2:])
                    labels_batch = torch.max(synthetic_labels[i], 1)[1]
                    loss = torch.nn.CrossEntropyLoss()(synthetic_model(inputs_batch), labels_batch)
                    loss.backward()
                optimizer.step()

        synthetic_model = deepcopy(self.model)
        synthetic_model.train()
        synthetic_optim = torch.optim.SGD(synthetic_model.parameters(), lr=eta_w, momentum=0.0)
        for i in range(20):
            synthetic_optim.zero_grad()
            inputs_batch = synthetic_inputs[i].view(-1, *synthetic_inputs.shape[2:])
            labels_batch = torch.max(synthetic_labels[i], 1)[1]
            loss = torch.nn.CrossEntropyLoss()(synthetic_model(inputs_batch), labels_batch)
            loss.backward()
            synthetic_optim.step()

        synthetic_W = {k: v for k, v in synthetic_model.named_parameters()}

        synthetic_gradients_flatten = torch.cat(
            [self.W[k].clone().flatten() - v.clone().flatten() for k, v in synthetic_W.items()])
        real_gradients = torch.cat([v.flatten() for v in deepcopy(self.dW).values()])
        cos = torch.sum(synthetic_gradients_flatten * real_gradients) / (
                    torch.norm(synthetic_gradients_flatten) * torch.norm(real_gradients) + 1e-12)
        scale_factor = cos * torch.norm(real_gradients) / torch.norm(synthetic_gradients_flatten)
        scale_factor = 0.0 if torch.isnan(scale_factor) else scale_factor.item()
        return deepcopy(synthetic_inputs), deepcopy(synthetic_labels), scale_factor, cos.item()

    def compute_synthetic_sample(self, n_sample, n_classes):
        synthetic_input_size = [n_sample] + list(next(iter(self.train_loader))[0].shape[1:])
        synthetic_inputs = torch.randn(tuple(synthetic_input_size), device=self.device, requires_grad=True)
        synthetic_labels = torch.randn((n_sample, n_classes), device=self.device, requires_grad=True)

        synthetic_model = deepcopy(self.model)
        synthetic_model.eval()
        optimizer = torch.optim.LBFGS([synthetic_inputs, synthetic_labels])
        best_inputs, best_labels, best_loss = synthetic_inputs.clone(), synthetic_labels.clone(), float("inf")
        s2 = torch.cat([v.clone().flatten() for v in self.dW_residual.values()])

        for _ in range(10):
            def closure():
                optimizer.zero_grad()
                inputs_batch = synthetic_inputs.view(-1, *synthetic_inputs.shape[1:])  # 合并 batch
                labels_batch = torch.max(synthetic_labels, 1)[1]
                synthetic_preds = synthetic_model(inputs_batch)
                loss = torch.nn.CrossEntropyLoss()(synthetic_preds, labels_batch)
                dy_dx = torch.autograd.grad(loss, synthetic_model.parameters(), create_graph=True, allow_unused=True)
                s1 = torch.cat([v.flatten() for v in dy_dx])
                grad_loss = 1.0 - torch.abs(torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12))
                grad_loss.backward()
                return grad_loss

            optimizer.step(closure)
            current_loss = closure()
            if 0 <= current_loss.item() < best_loss:
                best_inputs = synthetic_inputs.clone()
                best_labels = synthetic_labels.clone()
                best_loss = current_loss.item()

        inputs_batch = best_inputs.view(-1, *best_inputs.shape[1:])
        labels_batch = torch.max(best_labels, 1)[1]
        preds = synthetic_model(inputs_batch)
        loss = torch.nn.CrossEntropyLoss()(preds, labels_batch)
        synthetic_gradients = torch.autograd.grad(loss, synthetic_model.parameters(), create_graph=True)
        synthetic_gradients_flatten = torch.cat([v.clone().flatten() for v in synthetic_gradients])
        real_gradients = torch.cat([v.flatten() for v in deepcopy(self.dW_residual).values()])
        cos = torch.sum(synthetic_gradients_flatten * real_gradients) / (
                    torch.norm(synthetic_gradients_flatten) * torch.norm(real_gradients) + 1e-12)
        scale_factor = cos * torch.norm(real_gradients) / torch.norm(synthetic_gradients_flatten)
        scale_factor = 0.0 if torch.isnan(scale_factor) else scale_factor.item()

        synthetic_gradients_dict = {name: synthetic_gradients[i] * scale_factor for i, name in
                                    enumerate(self.dW_residual)}
        subtract_(self.dW_residual, self.dW_residual, synthetic_gradients_dict)
        return best_inputs, best_labels, scale_factor, cos.item()


class Server:
    def __init__(self, test_loader, device, model_fn=None):
        self.device = device
        self.model = model_fn().to(device)
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.clients = []
        self.test_loader = test_loader
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

    def evaluate(self):
        acc = eval_op(self.model, self.test_loader, self.device)
        print(f"Server Accuracy: {acc * 100:.2f}%")
        return acc

    def select_clients(self, frac=0.4):
        n = max(int(len(self.clients) * frac), 1)
        return random.sample(self.clients, n)

    def aggregate_synthetic_gradients(self, synthetics, scale_factors, dws, method="ours"):
        client_gradients = []
        comm_bytes_total = 0
        flops_total = 0

        for i, (inputs, labels) in enumerate(synthetics):
            if method == "fedsynth":
                batch_inputs = inputs.view(-1, *inputs.shape[2:]) 
                batch_labels = torch.argmax(labels.view(-1, labels.shape[-1]), dim=1)  
            else:
                batch_inputs = inputs  
                batch_labels = torch.argmax(labels, dim=1)  

            synthetic_model = deepcopy(self.model)
            preds = synthetic_model(batch_inputs)

            
            if preds.size(0) != batch_labels.size(0):
                raise ValueError(f"预测值和标签的批量大小不匹配: preds={preds.size(0)}, labels={batch_labels.size(0)}")

            
            loss = torch.nn.CrossEntropyLoss()(preds, batch_labels)
            gradients = torch.autograd.grad(loss, synthetic_model.parameters(), create_graph=True)
            gradients = [scale_factors[i] * g.clone().detach() for g in gradients]
            client_gradients.append({name: gradients[j] for j, name in enumerate(self.W)})


            if method == "fedsynth":
                bytes_inputs = inputs.numel() * inputs.element_size()
                bytes_labels = labels.numel() * labels.element_size()
                comm_bytes_total += bytes_inputs + bytes_labels

            elif method == "ours":
                input_bytes = batch_inputs.numel() * batch_inputs.element_size()
                label_bytes = labels.numel() * labels.element_size()
                scale_bytes = 4  
                comm_bytes_total += input_bytes + label_bytes + scale_bytes

            batch_size_for_flops = batch_inputs.size(0)
            flops_total += estimate_flops(self.model, input_size=(1, 28, 28), batch_size=batch_size_for_flops,device=self.device)

        for client in self.clients:
            num_batches = len(client.train_loader)
            flops_per_batch = estimate_flops(self.model, input_size=(1, 28, 28),batch_size=client.train_loader.batch_size, device=self.device)
            flops_total += num_batches * flops_per_batch * 5  

        reduce_add_average([self.W], client_gradients)
        self.history['communication_cost'].append(comm_bytes_total)
        self.history['computation_cost'].append(flops_total)
        self.history['total_comm_bytes'] += comm_bytes_total
        self.history['total_computation_flops'] += flops_total

    def train(self, communication_rounds=10, local_epochs=1, method="fedsynth", n_sample=10, n_classes=10):
        print("开始联邦学习训练...")
        start_time = time.time()
        for round_idx in range(1, communication_rounds + 1):
            print(f"\n=== 通信轮次 {round_idx}/{communication_rounds} ===")

            for client in self.clients:
                copy_state(client.W, self.W)

            model_bytes = sum([p.numel() * p.element_size() for p in self.W.values()])

            participating_clients = self.select_clients(frac=0.4)
            downlink_bytes_total = model_bytes*len(participating_clients)
            synthetics, scale_factors, losses = [], [], []

            for client in participating_clients:

                loss = client.compute_weight_update(epochs=local_epochs)
                losses.append(loss)
                client.reset()

                if method == "fedsynth":
                    inputs, labels, scale, _ = client.compute_fedsynth(n_sample, n_classes, eta_w=0.01, eta=0.01,
                                                                       epochs=local_epochs)
                else:
                    inputs, labels, scale, _ = client.compute_synthetic_sample(n_sample, n_classes)

                synthetics.append((inputs, labels))
                scale_factors.append(scale)

            self.aggregate_synthetic_gradients(synthetics, scale_factors, [c.dW for c in participating_clients],
                                               method=method)
            uplink_bytes_total = self.history['communication_cost'][-1]  

            total_bytes_this_round = uplink_bytes_total + downlink_bytes_total
            self.history['communication_cost'][-1] = total_bytes_this_round
            self.history['total_comm_bytes'] += downlink_bytes_total  

            
            acc = self.evaluate()
            round_time = time.time() - round_start

            self.history['accuracy'].append(acc * 100)
            self.history['loss'].append(np.mean(losses))
            self.history['communication_rounds'] = round_idx
            self.history['per_round_time'].append(round_time)

            print(f"轮次 {round_idx} loss={np.mean(losses):.4f}, "
                  f"上行={uplink_bytes_total} B, "
                  f"下行={downlink_bytes_total / 1024 ** 2:.4f} MB, "
                  f"总通信={total_bytes_this_round / 1024 ** 2:.4f} MB, "
                  f"耗时={round_time:.2f}s")

        self.history['total_training_time'] = time.time() - start_time
        print("联邦学习训练完成!")
        self.print_cost_summary()
        self.save_detailed_history_to_txt(self.history, "3sfc_mnist10.txt")


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
            f.write(
                f"平均每轮计算量: {history['total_computation_flops'] / history['communication_rounds']:.0f} FLOPs\n\n")

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


# =================== main ===================
def main():
    parser = argparse.ArgumentParser(description='联邦学习FedSynth/Ours实现')
    parser.add_argument('--num_clients', type=int, default=20, help='客户端数量')
    parser.add_argument('--communication_rounds', type=int, default=200, help='通信轮次')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮次')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--classes_per_client', type=int, default=3, help='每个客户端的类别数')
    parser.add_argument('--iid', action='store_true', default=False, help='使用IID数据分布')
    parser.add_argument('--method', type=str, default='fedsynth', choices=['fedsynth', 'ours'], help='联邦算法')
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
        from torch.utils.data import random_split
        lengths = [len(train_dataset)//args.num_clients]*args.num_clients
        remainder = len(train_dataset) - sum(lengths)
        for i in range(remainder):
            lengths[i] += 1
        client_datasets = random_split(train_dataset, lengths)
    else:
        client_datasets = create_non_iid_datasets(train_dataset, args.num_clients, args.classes_per_client)

    clients = []
    for i in range(args.num_clients):
        train_loader = DataLoader(client_datasets[i], batch_size=args.batch_size, shuffle=True)
        clients.append(Client(i, train_loader, device, lr=args.lr, model_fn=SimpleCNN, n_classes=10))


    server = Server(test_loader, device, model_fn=SimpleCNN)
    server.attach_clients(clients)

    print("初始全局模型性能:")
    server.evaluate()

    server.train(communication_rounds=args.communication_rounds,
                 local_epochs=args.local_epochs,
                 method=args.method,
                 n_sample=10,
                 n_classes=10)

    print("\n=== 所有轮的准确率 ===")
    for i, acc in enumerate(server.history['accuracy']):
        print(f"第{i}轮: {acc:.2f}%")

if __name__ == "__main__":
    main()
