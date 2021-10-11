import numpy as np
import torch
import faiss
from prototype import get_indices_sparse


"""
refer to: Authors: Wouter Van Gansbeke, Simon Vandenhende
"""

def compute_cluster_assignment(centroids, x):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel() 

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.zeros([self.n, self.dim],dtype=torch.float64)
        self.targets = torch.zeros(self.n,dtype=torch.int64)
        self.filenames = []
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred
    
    def get_knn(self,centroids, test_embeddings):
        d = centroids.shape[1]
        index = faiss.IndexFlatL2(d)
        if faiss.get_num_gpus()>0:
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(centroids)
        distances,indices = index.search(test_embeddings, self.k)
        neighbor_targets = np.take(self.targets, indices, axis=0) # Exclude sample itself for eval
        anchor_targets = np.repeat(self.targets.reshape(-1,1), self.k, axis=1)
        accuracy = np.mean(neighbor_targets == anchor_targets)
        return  accuracy


    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim) # inner product --> cosin simi
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices
    
    def cal_mean_rep(self,config):
        targets = self.targets
        if config.device=="cuda":
            emb_sums = torch.zeros(config.dataset.n_classes, config.model.features_dim).cuda(non_blocking=True)
            targets=targets.cpu().numpy()
        else:
            emb_sums = torch.zeros(config.dataset.n_classes, config.model.features_dim)
            targets=targets.numpy()
        where_helper = get_indices_sparse(targets)
        for k in range(len(where_helper)):
            # print("where_helper[k]: ",where_helper[k])
            if len(where_helper[k][0]) > 0:
                emb_sums[k] = torch.sum(
                                self.features[where_helper[k][0]],
                                dim=0,
                            )/len(where_helper[k][0])
        self.centroids = emb_sums.copy()

        return emb_sums


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
        


