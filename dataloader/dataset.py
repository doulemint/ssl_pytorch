import pathlib
from PIL import Image,ImageFile

import transforms 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import yacs,os,pickle
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

import torch,random
import torchvision
from torch.utils.data import Dataset

from transforms import create_transform


def get_files(root,mode,label_map_dir):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    else:
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: root + x, os.listdir(root)))
        # print("image_folders",image_folders)
        all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
        # print("all_images",all_images)
        if mode == "val":
            print("loading val dataset")
        elif mode == "train":
            print("loading train dataset")
        else:
            raise Exception("Only have mode train/val/test, please check !!!")
        if os.path.exists(label_map_dir):
            with open(label_map_dir, 'rb')  as f:
                label_dict=pickle.load(f)
            for file in tqdm(all_images):
                all_data_path.append(file)
                name=file.split(os.sep)[-2]
                labels.append(label_dict[name])
        else:
            label_dict={}
            for file in tqdm(all_images):
                all_data_path.append(file)
                name=file.split(os.sep)[-2] #['', 'data', 'nextcloud', 'dbc2017', 'files', 'images', 'train', 'Diego_Rivera', 'Diego_Rivera_21.jpg']
                # print(name)
                if name not in label_dict:
                    label_dict[name]=len(label_dict)
                labels.append(label_dict[name])
            pickle.dump(label_dict,open(label_map_dir, 'wb'))
            # labels.append(int(file.split(os.sep)[-2]))
        print(label_dict)
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        return all_files

def get_img2(imgsrc):
    img = np.asarray(Image.open(imgsrc).convert('RGB'))
    return img

class MyDataset(Dataset):
    def __init__(self, df, transforms=None,output_label=True):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.output_label=output_label
        
        if output_label == True:
            self.labels = self.df['label'].values
        # if soft==True:
        #     self.labels = np.identity(n_class)[self.labels].astype(np.float32)
        #     if label_smooth:
        #         self.labels = self.labels*(1 - epsilon)+ np.ones_like(self.labels) * epsilon / n_class

            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = get_img2("{}".format(self.df.loc[index]['filename']))
        # if self.is_df:       
        #     img  = get_img2("{}".format(self.df.loc[index]['file']))
        # else:
        #   if self.data_type=='wiki22':
        #         img = get_img2("{}/{}".format(self.data_root, self.df.loc[index]['image']))
        #   else:
        #     img  = get_img2("{}".format(self.df.loc[index]['filename']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.output_label == True:
            return img, target
        else:
            return img

class pesudoMyDataset(MyDataset):
    #do mixup and label guess 
    def __init__(self, df,unlabel_df, transforms=None, output_label=True,alpha=1):
        
        super(pesudoMyDataset, self).__init__(
            df=df, transforms=transforms,output_label=output_label)
        
        self.unlabel_df = unlabel_df
        self.transforms = transforms
        self.alpha = alpha
        #get where_helper only from labeled dataset
        self.where_helper = get_indices_sparse(self.labels)

        self.labels=self.unlabel_df['label'].values
    
    def __len__(self):
        return self.unlabel_df.shape[0]
        
    def __getitem__(self, index: int):
        lam = np.random.beta(self.alpha, self.alpha)
        target = self.labels[index]
        img_a  = get_img2("{}".format(self.unlabel_df.loc[index]['filename']))
        img=img_a

        if len(self.where_helper[target]) > 0:
            img_b = get_img2(self.df.loc[random.choice(self.where_helper[target])]['filename'])
            img = lam*img_a + (1-lam)*img_b

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, target

def create_pair_dataset(config,weakAugment:bool):
    #if not use_test_as_val:
    if weakAugment:
            transform = create_transform(config, is_train=False)
    else:
            transform = create_transform(config, is_train=True)

    dataset_dir = config.dataset.dataset_dir
    label_map_dir = config.train.output_dir+"/label_map.pkl"
    train_df=get_files(dataset_dir+"/train/","train",label_map_dir)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=0)
    for trn_idx, val_idx in sss.split(train_df['filename'], train_df['label']):
        train_ = train_df.loc[trn_idx,:].reset_index(drop=True)
        unlabeled_ = train_df.loc[val_idx,:].reset_index(drop=True)
    train_dataset=MyDataset(train_,transforms=transform)
    unlabled_dataset=MyDataset(unlabeled_,transforms=transform,output_label=False)
    return train_dataset, unlabled_dataset

#labeled data return normal dataset
def create_datasets(config: yacs.config.CfgNode,weakAugment:bool,labeled:bool,df=None)->Dataset:
    dataset_dir = config.dataset.dataset_dir
    label_map_dir = config.train.output_dir+"/label_map.pkl"
    if weakAugment:
            transform = create_transform(config, is_train=False)
    else:
            transform = create_transform(config, is_train=True)

    if df is not None:
        return MyDataset(df,transforms=transform,output_label=labeled)

    if labeled:
        #if use albumentation augmentation we need to use custome dataset
        if config.augmentation.use_albumentations:
            train_df=get_files(dataset_dir+"/train/","train",label_map_dir)
            train_dataset=MyDataset(train_df,transforms=transform)
        else:
        # or normal augmentation from torchvision
            train_dataset = torchvision.datasets.ImageFolder(
                    dataset_dir+"/train", transform=transform)
        return train_dataset
    else:
        #return unlabeled dataset
        if config.augmentation.use_albumentations:
            val_df=get_files(dataset_dir+"/val/","test",label_map_dir)
            val_dataset=MyDataset(val_df,transforms=transform,output_label=False)
        else:
            val_transform = create_transform(config, is_train=False)
            val_dataset = torchvision.datasets.ImageFolder(dataset_dir+"/val",
                                                            transform=transform)
        return val_dataset
    