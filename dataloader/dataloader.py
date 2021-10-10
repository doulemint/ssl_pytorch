
from .dataset import create_datasets,pesudoMyDataset,MyDataset,create_pair_dataset
import yacs,torch
import pandas as pd
from utils.prototype import get_indices_sparse
from torch.utils.data import DataLoader

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
    train_datset = create_datasets(config,isweak,False,df)
    train_loader = torch.utils.data.DataLoader(
        train_datset,
        batch_size=config.train.batch_size,
        pin_memory=config.train.dataloader.pin_memory,
        drop_last=False,
        shuffle=False,        
        num_workers=config.train.dataloader.num_workers,
    )
    return train_loader

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

def create_fewshot_dataloader(config: yacs.config.CfgNode,isweak:bool,unlabeled_df:pd.DataFrame)->DataLoader:
    train_Dataset, unlabeled_Dataset = create_pair_dataset(config,isweak,True)
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
    r
    return train_loader,unlabeled_loader

def create_dataloader(config: yacs.config.CfgNode,isweak:bool,islabeled:bool,use_test_as_val:bool):
    if not use_test_as_val:
        return create_fewshot_dataloader()
    if islabeled:
        return create_labaled_dataloader(config,isweak)
    else:
        return create_unlabaled_dataloader(config,isweak)

    
    