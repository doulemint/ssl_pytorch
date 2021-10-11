
from .dataset import create_datasets,pesudoMyDataset,MyDataset,create_pair_dataset
from typing import Tuple, Union
import yacs,torch
import pandas as pd
from utils.prototype import get_indices_sparse
from torch.utils.data import DataLoader

def mine_nn(memory_bank_unlabeled,num_neighbors):
    _, pred = memory_bank_unlabeled.features[:memory_bank_unlabeled.ptr].topk(num_neighbors, 0, False, True)
    pred = pred.t()
    topk_files=[]
    topk_labels=[]
    for labels,index in enumerate(pred):
        topk_files.extend([memory_bank_unlabeled.filenames[i] for i in index])
        topk_labels.extend([labels]*num_neighbors)
    unlabeled_df = pd.DataFrame({"filename": topk_files, "label": topk_labels})
    return unlabeled_df

def get_mixup(config,memory_bank_unlabeled,num_neighbors):
    #mine the nearest nearbor
    #convert topk to df
    # pred size(num_neighbors x n_classes)
    unlabeled_df = mine_nn(memory_bank_unlabeled,num_neighbors)
    
    # prototype_mixture update new embeddings
    Mixup_dataloader = create_mixup_dataloader(config,True,unlabeled_df)
    # emb_sums = predefined_prototype(config,model,Mixup_dataloader)
    return Mixup_dataloader 

def create_labaled_dataloader(config: yacs.config.CfgNode,isweak:bool)->DataLoader:
    train_datset = create_datasets(config,isweak,True)
    train_loader = torch.utils.data.DataLoader(
        train_datset,
        batch_size=config.train.batch_size,
        pin_memory=config.train.dataloader.pin_memory,
        drop_last=False,
        shuffle=True,        
        num_workers=config.train.dataloader.num_workers,
    )
    return train_loader

def create_constrastive_dataloader(config: yacs.config.CfgNode,isweak:bool,df:pd.DataFrame)->DataLoader:
    train_datset = create_datasets(config,isweak,True,df)
    train_loader = torch.utils.data.DataLoader(
        train_datset,
        batch_size=config.train.batch_size,
        pin_memory=config.train.dataloader.pin_memory,
        drop_last=False,
        shuffle=True,        
        num_workers=config.train.dataloader.num_workers,
    )
    return train_loader

def get_positive_sample(config,memory_bank_unlabeled,num_neighbors,mask):
    unlabeled_df = mine_nn(memory_bank_unlabeled,num_neighbors)
    constrastive_dataloader = create_constrastive_dataloader(config,True,unlabeled_df)
    return constrastive_dataloader

def create_unlabaled_dataloader(config: yacs.config.CfgNode,isweak:bool)->DataLoader:
    val_datset = create_datasets(config,isweak,False)
    val_loader = torch.utils.data.DataLoader(
        val_datset,
        batch_size=config.train.batch_size,
        pin_memory=config.train.dataloader.pin_memory,
        drop_last=False,
        shuffle=True,        
        num_workers=config.train.dataloader.num_workers,
    )
    return val_loader

def create_mixup_dataloader(config: yacs.config.CfgNode,isweak:bool,unlabeled_df:pd.DataFrame)->DataLoader:
    MyDataset = create_datasets(config,isweak,True)
    mixup_datset = pesudoMyDataset(MyDataset.df,unlabeled_df,MyDataset.transforms,alpha=1)
    mixup_loader = torch.utils.data.DataLoader(
        mixup_datset,
        batch_size=config.train.batch_size,
        pin_memory=config.train.dataloader.pin_memory,
        drop_last=False,
        shuffle=True,        
        num_workers=config.train.dataloader.num_workers,
    )
    return mixup_loader

def create_fewshot_dataloader(config: yacs.config.CfgNode,isweak:bool)->Union[Tuple[DataLoader, DataLoader], DataLoader]:
    train_Dataset, unlabeled_Dataset = create_pair_dataset(config,isweak)
    train_loader = torch.utils.data.DataLoader(
        train_Dataset,
        batch_size=config.train.batch_size,
        pin_memory=config.train.dataloader.pin_memory,
        drop_last=False,
        shuffle=True,        
        num_workers=config.train.dataloader.num_workers,
    )
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_Dataset,
        batch_size=config.train.batch_size,
        pin_memory=config.train.dataloader.pin_memory,
        drop_last=False,
        shuffle=True,        
        num_workers=config.train.dataloader.num_workers,
    )
    
    return train_loader,unlabeled_loader

def create_dataloader(config: yacs.config.CfgNode,isweak:bool,islabeled:bool,use_test_as_val:bool):
    if not use_test_as_val:
        return create_fewshot_dataloader(config,isweak)
    if islabeled:
        return create_labaled_dataloader(config,isweak)
    else:
        return create_unlabaled_dataloader(config,isweak)

    
    