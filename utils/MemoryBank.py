import numpy as np
import torch

"""
refer to: Authors: Wouter Van Gansbeke, Simon Vandenhende
"""

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b
        
    def reset(self):
        self.ptr = 0 

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


if __name__ == '__main__':
    import random
    features_dim=10
    n_classes=5
    temperature=0.9
    length=10
    loader=[ (random.randint(n_classes),random.Random(features_dim)) for i in range(length)]
    memory_bank_base = MemoryBank(len(loader), 
                                features_dim,n_classes, temperature)
    memory_bank_base.reset()

    for i, batch in enumerate(loader):
        memory_bank_base.update(batch, i)
    
    #算mean 根据 targets cluster
