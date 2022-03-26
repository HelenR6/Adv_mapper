from dataset import CustomImageDataset
from base import CustomResNet
from mapper import RegMapper
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import h5py
import json
import numpy as np
from base import CustomResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
    ])
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
      
        yield iterable[ndx:min(ndx + n, l)]
dataset = CustomImageDataset('s_stretch_session1',preprocess,'train')
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ])
mapper=RegMapper(arch='st_resnet',session='s_stretch_session1',preprocess=preprocess)