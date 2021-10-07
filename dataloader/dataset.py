import pathlib
from PIL import Image,ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import yacs
import torch
import torchvision

from transformers import create_transform





#labeled data return normal dataset
def create_datasets(config: yacs.config.CfgNode,weakAugment:bool,labeled:bool):
    dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
    if weakAugment:
            val_transform = create_transform(config, is_train=False)
    else:
            train_transform = create_transform(config, is_train=True)
    if labeled:
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_dir / 'train', transform=train_transform)
        return train_dataset
    else:
        if weakAugment:
            val_transform = create_transform(config, is_train=False)
            val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                            transform=val_transform)
        return val_dataset
    