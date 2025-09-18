import torch

class STCCompressor:
    def __init__(self, sparsity=0.01):
        
        self.sparsity = sparsity
        self.indices = None
        self.compressed = None
        self.mean_val = None
        self.total_len = None

    def compress(self, tensor):
      
        tensor = tensor.view(-1)
        self.total_len = tensor.numel()
        k = max(1, int(self.sparsity * self.total_len))

        values, indices = torch.topk(tensor.abs(), k)
        mask = torch.zeros_like(tensor)
        mask[indices] = 1

        selected = tensor * mask
        self.mean_val = values.mean() if values.numel() > 0 else torch.tensor(0.0, device=tensor.device)
        self.compressed = torch.sign(selected[indices]) * self.mean_val
        self.indices = indices

        return self.indices.cpu(), self.compressed.cpu(), self.mean_val.cpu(), self.total_len

    def decompress(self, indices=None, compressed=None, mean_val=None, total_len=None):
        
        if indices is None:
            indices = self.indices
        if compressed is None:
            compressed = self.compressed
        if mean_val is None:
            mean_val = self.mean_val
        if total_len is None:
            total_len = self.total_len

        tensor = torch.zeros(total_len)
        if len(indices) > 0:
            tensor[indices] = compressed
        return tensor
