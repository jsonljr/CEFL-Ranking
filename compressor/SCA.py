import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import argparse





class SCACompressor:


    def __init__(self, sparsity_fraction=0.1):
  
        self.sparsity_fraction = sparsity_fraction

    def compress(self, delta):
  
        if isinstance(delta, dict):
            compressed_delta = {}
            for key, tensor in delta.items():
                compressed_delta[key] = self._compress_tensor(tensor)
            return compressed_delta
        else:
            return self._compress_tensor(delta)

    def _compress_tensor(self, tensor):

        flat_tensor = tensor.flatten()
        n_elements = flat_tensor.numel()
        k = max(1, int(n_elements * self.sparsity_fraction))

 
        positive_values, positive_indices = torch.topk(flat_tensor.clamp(min=0), k)
        negative_values, negative_indices = torch.topk(-flat_tensor.clamp(max=0), k)
        negative_values = -negative_values  

        pos_mean = positive_values.mean() if positive_values.numel() > 0 else torch.tensor(0.0)
        neg_mean = negative_values.mean() if negative_values.numel() > 0 else torch.tensor(0.0)

        if pos_mean >= torch.abs(neg_mean) and positive_values.numel() > 0:
            values = torch.full((positive_indices.numel(),), pos_mean, device=tensor.device)
            indices = positive_indices
        elif negative_values.numel() > 0:
            values = torch.full((negative_indices.numel(),), neg_mean, device=tensor.device)
            indices = negative_indices
        else:
            values = torch.tensor([0.0], device=tensor.device)
            indices = torch.tensor([0], device=tensor.device)

        return (values, indices)

