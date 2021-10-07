import yacs
from typing import Callable, Tuple

import numpy as np
import torchvision
import yacs.config

from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomBrightnessContrast, Perspective, CLAHE, 
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, ColorJitter, GaussNoise, MotionBlur, MedianBlur,
    Emboss, Sharpen, Flip, OneOf, SomeOf, Compose, Normalize, CoarseDropout, CenterCrop, GridDropout, Resize
)
from albumentations.pytorch import ToTensorV2

def _get_dataset_stats(
        config: yacs.config.CfgNode) -> Tuple[np.ndarray, np.ndarray]:
    name = config.dataset.name
    if name == 'CIFAR10':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    elif name == 'CIFAR100':
        # RGB
        mean = np.array([0.5071, 0.4865, 0.4409])
        std = np.array([0.2673, 0.2564, 0.2762])
    elif name == 'MNIST':
        mean = np.array([0.1307])
        std = np.array([0.3081])
    elif name == 'FashionMNIST':
        mean = np.array([0.2860])
        std = np.array([0.3530])
    elif name == 'KMNIST':
        mean = np.array([0.1904])
        std = np.array([0.3475])
    elif name == 'ImageNet':
        # RGB
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError()
    return mean, std


def create_transform(config: yacs.config.CfgNode,
                              is_train: bool) -> Callable:
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_albumentations:
            transforms = [
            OneOf([
            CoarseDropout(p=0.5),
            GaussNoise(),
            ], p=0.5),
            SomeOf([
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            ], n=3, p=0.6),
            ]
            # if config.augmentation.use_step_crop:
            #     transforms.append(
            #         OneOf([StepcropAlbu(p=0.5),StepcropAlbu(p=0.7,n=8,pos=1),StepcropAlbu(p=0.7,n=8,pos=2),StepcropAlbu(p=0.7,n=8,pos=3)]))
                    #OneOf([CornerCrop(p=1),StepcropAlbu(p=0.5),StepcropAlbu(p=0.7,n=16),StepcropAlbu(p=0.8,n=4)]))#
            transforms.extend([Resize(config.dataset.image_size, config.dataset.image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),])
            return Compose(transforms, p=1.0)
    else:
        if config.augmentation.use_albumentations:
            return Compose([
            Resize(config.dataset.image_size, config.dataset.image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),], p=1.)
       