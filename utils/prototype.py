from dataloader.dataloader import create_dataloader, create_mixup_dataloader
from dataloader.dataset import  create_datasets
from MemoryBank import MemoryBank
from models import build_model,get_encoder
import numpy as np
import pandas as pd
import pathlib,random

import torchvision, torch
from torch import nn
import torch.nn.functional as F

from scipy import spatial
from scipy.sparse import csr_matrix

def get_indices_sparse(data):
    """
    Is faster than np.argwhere. Used in loss functions like swav loss, etc
    """
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


def fill_memory_bank(loader,model,memory_bank,device):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        if device=="cuda":
            images = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
        else:
            images,targets=batch
        output,_ = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))

#refer to google research
# distribution alginment  
class PMovingAverage:
    def __init__(self, nclass, buf_size):
        # to do: how to use it in distribute system setting 
        # variable updates across shards
        self.ma = torch.ones([buf_size, nclass]) / nclass

    def __call__(self):
        v = torch.mean(self.ma,0)
        return v / torch.sum(v)

    def update(self, entry):
        entry = torch.mean(entry, 0)
        self.ma = torch.cat([self.ma[1:], [entry]])
        # return self.ma.clone()


# class PData:
#     def __init__(self, dataset: torchvision.datasets):
#         self.has_update = False
#         if dataset is not None:
#             self.p_data = tf.constant(dataset.p_unlabeled, name='p_data')
#         else:
#             # MEAN aggregation is used by DistributionStrategy to aggregate
#             # variable updates across shards
#             self.p_data = renorm(tf.ones([dataset.nclass]))
#             self.has_update = True

#     def __call__(self):
#         return self.p_data / tf.reduce_sum(self.p_data)

#     def update(self, entry, decay=0.999):
#         entry = tf.reduce_mean(entry, axis=0)
#         return tf.assign(self.p_data, self.p_data * decay + entry * (1 - decay))
#reproduce Semi-supervised Contrastive Learning with Similarity Co-calibration
#similarity distribution
def predefined_prototype(config,model,labalbed_dataloader):
    #weak augmentation, use label
    
    memory_bank_base = MemoryBank(len(labalbed_dataloader), 
                                config.model.feature_dim,
                                config.dataset.n_classes, config['criterion_kwargs']['temperature'])

    if config.device=="cuda":
        memory_bank_base.cuda()
    # Fill memory bank
    print('Fill memory bank for kNN...')
    fill_memory_bank(labalbed_dataloader, model, memory_bank_base,config.device)

    targets = memory_bank_base.targets
    if config.device=="cuda":
        emb_sums = torch.zeros(config.dataset.n_classes, config.model.feature_dim).cuda(non_blocking=True)
    else:
        emb_sums = torch.zeros(config.dataset.n_classes, config.model.feature_dim)
    where_helper = get_indices_sparse(targets.numpy())
    for k in range(len(where_helper)):
        print("where_helper[k]: ",where_helper[k])
        if len(where_helper[k][0]) > 0:
            emb_sums[k] = torch.sum(
                            memory_bank_base.features[where_helper[k][0]],
                            dim=0,
                        )/len(where_helper[k][0])

    #calcuate the prefined_prototype 

    return emb_sums

#we didn't use it, it was incorporated in label_assignment 
def similarity_distribution(config):
    unlabeled_dataloader=create_dataloader(config,True,False)
    memory_bank_unlabeled = MemoryBank(len(unlabeled_dataloader), 
                                config.dataset.n_classes,
                                config.dataset.n_classes, config.criterion_kwargs.temperature)
    
    model = build_model(config)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    emb_sums = predefined_prototype(config,model) 

    for batch in unlabeled_dataloader:
        feature_vector,_=model(batch)
        #calculate similarity
        p_s=torch.zeros(feature_vector.size(0), config.dataset.n_classes)
        #if batch_size>0 we have to calculate them one by one
        if len(feature_vector.size())>=2:
            for i,fv in enumerate(feature_vector):
                fv = torch.unsqueeze(fv,dim=0)
                p_s[i] = cos(fv,emb_sums)
        else:
            p_s[0] = cos(feature_vector,emb_sums)
        memory_bank_unlabeled.update(p_s,None)

    return memory_bank_unlabeled
    
#
# sharpen(label guess x similarity distribution)




        


def prototype_mixture(config):
    #build labeled dataset distribution
    #change it to dataset?
    labalbed_dataloader = create_dataloader(config,False,True)
    p_data=PData(labalbed_dataloader)
    #build unlabeled distribution
    unlabeled_dataloader=create_dataloader(config,False,False)
    p_model=PMovingAverage(unlabeled_dataloader)

    emb_sums,memory_bank_unlabeled=label_assignment(config,p_model)
    # p_model.update(memory_bank_unlabeled.features)

    return




    

if __name__ == '__main__':

    from utils.common_utils import create_config
    import argparse
    # Parser
    parser = argparse.ArgumentParser(description='ssl_base_model')
    parser.add_argument('--config',
                        help='Config file for the environment')
    args = parser.parse_args()
    config=create_config(args.config)
    output_dir = pathlib.Path(config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    model = get_encoder(config)
    predefined_prototype(config,model)