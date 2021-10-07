
from .dataset import create_datasets
import yacs,torch
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

def create_dataloader(config: yacs.config.CfgNode,isweak:bool,islabeled:bool):
    if islabeled:
        return create_labaled_dataloader(config,isweak)
    
    