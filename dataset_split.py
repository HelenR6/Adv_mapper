import os
import pandas as pd
import h5py
import numpy as np
from PIL import Image
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
class CustomImageDataset(Dataset):
    def __init__(self, session=None,transform=None,mode=None):
        self.transform = transform
        session_path=session.replace('_','/')
        final_path=session_path[:-1]+'_'+session_path[-1:]
        f = h5py.File('/content/gdrive/MyDrive/npc_v4_data.h5','r')
        temp=np.empty([640,1])
        kfold = KFold(n_splits=5, shuffle=True,random_state=42)
        train_ids=[]
        test_ids=[]
        for fold,(train_ids,test_ids) in enumerate(kfold.split(temp)):
            if fold==0:
                train_ids=train_ids
                test_ids=test_ids
        if mode=='train':
          image_data = f['images/naturalistic'][train_ids]
        if mode=='val_id':
          image_data = f['images/naturalistic'][test_ids]
        if mode=='val_ood':
          image_data=f['images/synthetic/monkey_'+final_path][:]
        x = np.array([np.array(self.transform((Image.fromarray(i)).convert('RGB'))) for i in image_data])
        self.images=torch.tensor(x)
        if mode=='train':
          n1 = f.get('neural/naturalistic/monkey_'+final_path)[:]
          self.target=np.mean(n1, axis=0)[train_ids]
        if mode=='val_id':
          n1 = f.get('neural/naturalistic/monkey_'+final_path)[:]
          self.target=np.mean(n1, axis=0)[test_ids]
        if mode=='val_ood':
          n1 = f.get('neural/synthetic/monkey_'+final_path)[:]
          self.target=np.mean(n1, axis=0)
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx,:]
        label=self.target[idx,:]
        return image, label
