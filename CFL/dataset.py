import numpy as np
import torch
import random
from torch.utils.data import Subset

def create_non_iid_datasets(dataset, num_clients, classes_per_client=2, min_per_class=10):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for idxs in class_indices:
        np.random.shuffle(idxs)
    client_indices = [[] for _ in range(num_clients)]
    class_ptr = np.zeros(num_classes, dtype=int)
    for client_id in range(num_clients):
        for j in range(classes_per_client):
            assigned_class = (client_id + j) % num_classes
            start = class_ptr[assigned_class]
            end = min(start + min_per_class, len(class_indices[assigned_class]))
            client_indices[client_id].extend(class_indices[assigned_class][start:end].tolist())
            class_ptr[assigned_class] = end
    props = np.random.lognormal(mean=0, sigma=2.0, size=(num_classes, num_clients, classes_per_client))
    remaining_per_class = [len(v) - class_ptr[i] for i, v in enumerate(class_indices)]
    props = np.array([[[remaining_per_class[c]]] for c in range(num_classes)]) * props / np.sum(props, (1, 2), keepdims=True)
    for client_id in range(num_clients):
        for j in range(classes_per_client):
            assigned_class = (client_id + j) % num_classes
            num_samples = int(props[assigned_class, client_id, j])
            num_samples += random.randint(300, 600)
            if num_clients <= 20:
                num_samples *= 2
            start = class_ptr[assigned_class]
            end = min(start + num_samples, len(class_indices[assigned_class]))
            client_indices[client_id].extend(class_indices[assigned_class][start:end].tolist())
            class_ptr[assigned_class] = end
    client_datasets = []
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        client_datasets.append(Subset(dataset, client_indices[i]))
        unique_classes = list(set(labels[client_indices[i]]))
        print(f"客户端 {i}: {len(client_indices[i])} 个样本, 类别: {unique_classes}")
    return client_datasets

def create_non_IID_datasets(dataset, num_clients, classes_per_client=2, min_per_class=10):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for idxs in class_indices:
        np.random.shuffle(idxs)
    client_indices = [[] for _ in range(num_clients)]
    class_ptr = np.zeros(num_classes, dtype=int)
    for client_id in range(num_clients):
        for j in range(classes_per_client):
            assigned_class = (client_id + j) % num_classes
            start = class_ptr[assigned_class]
            end = min(start + min_per_class, len(class_indices[assigned_class]))
            client_indices[client_id].extend(class_indices[assigned_class][start:end].tolist())
            class_ptr[assigned_class] = end
    props = np.random.lognormal(mean=0, sigma=2.0, size=(num_classes, num_clients, classes_per_client))
    remaining_per_class = [len(v) - class_ptr[i] for i, v in enumerate(class_indices)]
    props = np.array([[[remaining_per_class[c]]] for c in range(num_classes)]) * props / np.sum(props, (1, 2), keepdims=True)
    for client_id in range(num_clients):
        for j in range(classes_per_client):
            assigned_class = (client_id + j) % num_classes
            num_samples = int(props[assigned_class, client_id, j])
            num_samples += random.randint(300, 600)
            if num_clients <= 20:
                num_samples *= 2
            start = class_ptr[assigned_class]
            end = min(start + num_samples, len(class_indices[assigned_class]))
            client_indices[client_id].extend(class_indices[assigned_class][start:end].tolist())
            class_ptr[assigned_class] = end
    client_datasets = []
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        client_datasets.append(Subset(dataset, client_indices[i]))
        unique_classes = list(set(labels[client_indices[i]]))
        print(f"客户端 {i}: {len(client_indices[i])} 个样本, 类别: {unique_classes}")
    return client_datasets

def create_non_IID_datasets_strict(dataset, num_clients, classes_per_client=2, min_per_class=10):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for idxs in class_indices:
        np.random.shuffle(idxs)
    client_indices = [[] for _ in range(num_clients)]
    class_ptr = np.zeros(num_classes, dtype=int)
    client_classes = [set() for _ in range(num_clients)]
    for c in range(num_classes):
        available_clients = [i for i in range(num_clients) if len(client_classes[i]) < classes_per_client]
        client_id = np.random.choice(available_clients)
        client_classes[client_id].add(c)
        start = class_ptr[c]
        end = min(start + min_per_class, len(class_indices[c]))
        client_indices[client_id].extend(class_indices[c][start:end].tolist())
        class_ptr[c] = end
    for client_id in range(num_clients):
        while len(client_classes[client_id]) < classes_per_client:
            available_classes = [c for c in range(num_classes) if c not in client_classes[client_id]]
            chosen_class = np.random.choice(available_classes)
            client_classes[client_id].add(chosen_class)
            start = class_ptr[chosen_class]
            end = min(start + min_per_class, len(class_indices[chosen_class]))
            client_indices[client_id].extend(class_indices[chosen_class][start:end].tolist())
            class_ptr[chosen_class] = end
    for c in range(num_classes):
        remaining = len(class_indices[c]) - class_ptr[c]
        if remaining <= 0:
            continue
        clients_with_c = [i for i in range(num_clients) if c in client_classes[i]]
        props = np.random.lognormal(mean=0, sigma=2.0, size=len(clients_with_c))
        props = props / props.sum()
        for idx, client_id in enumerate(clients_with_c):
            num_samples = int(props[idx] * remaining)
            start = class_ptr[c]
            end = min(start + num_samples, len(class_indices[c]))
            client_indices[client_id].extend(class_indices[c][start:end].tolist())
            class_ptr[c] = end
    client_datasets = []
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        client_datasets.append(Subset(dataset, client_indices[i]))
        unique_classes = sorted(list(set(labels[client_indices[i]])))
        print(f"客户端 {i}: {len(client_indices[i])} 个样本, 类别: {unique_classes}")
    return client_datasets
