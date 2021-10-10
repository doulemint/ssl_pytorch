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
        self.filenames = []
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def update(self, features, targets,files=None):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        if files:
            self.filenames.extend(files)
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
    loader=[ (torch.randint(n_classes,(1,)),torch.unsqueeze(torch.from_numpy(np.random.random_sample(features_dim)),dim=0)) for i in range(length)]
    print("train data")
    print(loader)
    memory_bank_base = MemoryBank(len(loader), 
                    features_dim,n_classes, temperature)
    memory_bank_base.reset()

    for (i, batch) in loader:
        memory_bank_base.update(batch, i)
    
    #算mean 根据 targets cluster
    targets = memory_bank_base.targets
    emb_sums = torch.zeros(n_classes, features_dim)
    from.prototype import get_indices_sparse
    where_helper = get_indices_sparse(targets.numpy())
    for k in range(len(where_helper)):
        print("where_helper[",k,"]: ",where_helper[k])
        if len(where_helper[k][0]) > 0:
            emb_sums[k] = torch.sum(
                            memory_bank_base.features[where_helper[k][0]],
                            dim=0,
                        )/len(where_helper[k][0])
    print("predefined prototype")
    print(emb_sums)
    unlabeled_dataloader=[ torch.unsqueeze(torch.from_numpy(np.random.random_sample(features_dim)),dim=0) for i in range(5)]
    print(unlabeled_dataloader)
    memory_bank_unlabeled = MemoryBank(len(unlabeled_dataloader), 
                                n_classes,n_classes, temperature)    #calculate similarity
    print("similarity score")
    from torch import nn
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for feature_vector in unlabeled_dataloader:
        #calculate similarity
        p_s = cos(feature_vector,emb_sums)
        # print(p_s.size())
        p_s=torch.unsqueeze(p_s,dim=0)
        # print(p_s.size())
        memory_bank_unlabeled.update(p_s,torch.tensor(0))
    
    #mine the nearest nearbor
    sim = memory_bank_unlabeled.features
    print("sim: ",sim)
    num_neighbors = 3
    _, pred = sim.topk(num_neighbors, 0, True, True)
    # pred size(num_neighbors x n_classes)
    print(pred)
        


