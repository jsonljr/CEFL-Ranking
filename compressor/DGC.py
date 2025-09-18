import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from collections import OrderedDict

class DGCCompressor:

    def __init__(self, compression_rate=0.99, momentum_factor=0.9, residual=True):
        self.compression_rate = compression_rate
        self.momentum_factor = momentum_factor
        self.residual = residual
        self.momentum = None
        self.residual_grad = None
        self.mask = None
        self.last_flops_compressor = 0  

    def compress(self, gradient, model=None):
        if self.momentum is None:
            self.momentum = OrderedDict()
            self.residual_grad = OrderedDict()
            for name, param in model.named_parameters():
                self.momentum[name] = torch.zeros_like(param.data)
                self.residual_grad[name] = torch.zeros_like(param.data)

        compressed_grad = OrderedDict()
        self.mask = OrderedDict()
        self.last_flops_compressor = 0  

        for name, grad in gradient.items():
            if self.residual:
                grad = grad + self.residual_grad[name]
                self.last_flops_compressor += grad.numel()

            self.momentum[name] = self.momentum_factor * self.momentum[name] + grad
        
            self.last_flops_compressor += grad.numel() * 2
            grad_with_momentum = self.momentum[name]

    
            threshold = self._calculate_threshold(grad_with_momentum)
            self.last_flops_compressor += grad_with_momentum.numel()

            mask = (torch.abs(grad_with_momentum) > threshold).float()
            self.mask[name] = mask
            self.last_flops_compressor += grad_with_momentum.numel()

            compressed_grad[name] = grad_with_momentum * mask

            if self.residual:
                self.residual_grad[name] = grad - compressed_grad[name]
                self.last_flops_compressor += grad.numel()

        return compressed_grad

    def _calculate_threshold(self, tensor):
        total_params = tensor.numel()
        keep_num = int(total_params * (1 - self.compression_rate))
        if keep_num == 0:
            return torch.max(torch.abs(tensor)) + 1.0
        abs_values = torch.abs(tensor).view(-1)
        if keep_num >= total_params:
            return 0.0
        topk_values, _ = torch.topk(abs_values, keep_num, largest=True, sorted=True)
        if keep_num > 0:
            threshold = topk_values[-1]
        else:
            threshold = torch.tensor(0.0)
        self.last_flops_compressor += 2 * total_params  
        return threshold

    def get_mask(self):
        return self.mask

    def reset(self):
        self.momentum = None
        self.residual_grad = None
        self.mask = None
        self.last_flops_compressor = 0
