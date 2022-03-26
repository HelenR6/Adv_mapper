from dataset import CustomImageDataset
from base import CustomResNet
from mapper import RegMapper
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import h5py
import json
import torch
import numpy as np
from advertorch.attacks import LinfPGDAttack, L2PGDAttack,L1PGDAttack
from base import CustomResNet
import torch.nn as nn

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
adversary = L2PGDAttack(
mapper, loss_fn=nn.MSELoss(reduction="sum"), eps=14.2737,
nb_iter=20, eps_iter=1.784, rand_init=True, clip_min=-2.1179, clip_max=2.6400,
targeted=False)
for i, (images, target) in enumerate(data_loader):
    # print(images.shape)
    print("target shape")
    print(target.shape)
    # mapper.forward(images.double())
    adv_untargeted = adversary.perturb(images.double(), target.double())
    # mapper.forward(adv_untargeted.double())
    diff=adv_untargeted-images
    print(diff.shape)
    diff_norm=torch.norm(diff,p=2,dim=(1,2,3))
    print(diff_norm)
