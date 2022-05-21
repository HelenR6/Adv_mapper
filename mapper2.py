import h5py
import numpy as np
from PIL import Image
import os.path
from os import path
from base import CustomResNet
import torch
import torch.nn as nn
import json
import torchvision.models as models
from sklearn.decomposition import PCA

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
      
        yield iterable[ndx:min(ndx + n, l)]
class RegMapper(nn.Module):
    def __init__(self, arch=None, best_layer=None,input_images=None,neuron_target=None,session=None,preprocess=None):
        super(RegMapper, self).__init__()
        self.arch=arch
        with open(f'/content/gdrive/MyDrive/V4/{session}/st_resnet_natural_mean.json') as json_file:
          load_data = json.load(json_file)
          json_acceptable_string = load_data.replace("'", "\"")
          d = json.loads(json_acceptable_string)
          # get the layer with the highest ID neural prediction. 
          self.best_layer=max(d, key=d.get)
        self.layer_list=layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
        self.input_images=input_images
        self.neuron_target=neuron_target
        self.session=session
        self.preprocess=preprocess
        
        self.read_data()
        self.attach_pca()
        self.attach_reg()
        self.double()
        
    def get_activation(self,name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    def load_model(self):
        if self.arch=="st_resnet":
            # load checkpoint for st resnet
            resnet=models.resnet50(pretrained=True)


        return resnet
    def truncate(self):
        model=self.load_model()
        truncated_nn=CustomResNet(model=model,best_layer=self.best_layer,layer_list=self.layer_list)
        return truncated_nn.part
    def extract_features(self):
      counter=0
      for minibatch in batch(self.input_images,64):
        output=self.features(minibatch)
        if counter==0:
          with h5py.File('st_resnet_natural_layer_activation.hdf5','w')as f:
              dset=f.create_dataset(self.best_layer,data=output.cpu().detach().numpy())
        else:
          with h5py.File('st_resnet_natural_layer_activation.hdf5','r+')as f:
              data = f[self.best_layer]
              a=data[...]
              print(a.shape)
              del f[self.best_layer]
              dset=f.create_dataset(self.best_layer,data=np.concatenate((a,output.cpu().detach().numpy()),axis=0))
        counter=counter+1

    def attach_pca(self):
        # print("attach pca")
        # truncate the model (only keep layers before the best layer)
        self.features=self.truncate()
        # compute the PCA weight based on the offline features 
        pca_components=self.offline_pca()
        print("shapes!!!!!!!!!")
        print(pca_components.shape)
        self.features.add_module('flatten',torch.nn.Flatten(start_dim=1))
        pca_layer=torch.nn.Linear(in_features=pca_components.shape[1], out_features=pca_components.shape[0], bias=False)
        # initialize PCA layer with offline PCA weights
        # pca_layer.data=torch.FloatTensor(pca_components)
        pca_layer.weight=torch.nn.Parameter(torch.FloatTensor(pca_components.transpose()))
        self.features.add_module('pca',pca_layer)
        # freeze network layers and PCA layer
        for param in self.features.parameters():
          param.requires_grad = False
    def attach_reg(self):
        print("attach reg")
        # attach linear mapping layer. 
        # self.features.add_module('reg',torch.nn.Linear(in_features=self.input_images.shape[0], out_features=self.neuron_target.shape[1], bias=False))
        self.features.add_module('reg',torch.nn.Linear(in_features=640, out_features=52, bias=False))
    def read_data(self):
        session_path=self.session.replace('_','/')
        final_path=session_path[:-1]+'_'+session_path[-1:]
        print(final_path)
        f = h5py.File('/content/gdrive/MyDrive/npc_v4_data.h5','r')
        natural_data = f['images/naturalistic'][:]
        
        x = np.array([np.array(self.preprocess((Image.fromarray(i)).convert('RGB'))) for i in natural_data])
        self.input_images=torch.tensor(x)
        print(self.input_images.shape)

        n1 = f.get('neural/naturalistic/monkey_'+final_path)[:]
        self.neuron_target=np.mean(n1, axis=0)
        print(self.neuron_target.shape)
    
        return None
    def offline_pca(self):
        if not path.exists('st_resnet_natural_layer_activation.hdf5'):
          # extract features from the best layer. 
          self.extract_features()
        with h5py.File(f'st_resnet_natural_layer_activation.hdf5','r')as f:
          a=f[self.best_layer][...]
 
        
        if not path.exists(f'{self.session}_pca.npy'):
          # compute PCA weights
          pca=PCA(random_state=1)
          pca.fit_transform(torch.tensor(a).clone().detach().cpu().reshape(640,-1))
          pca_features=pca.components_.transpose()
          np.save(f'{self.session}_pca.npy',pca_features)
        else:
          # load previously saved PCA weights
          pca_features=np.load(f'gdrive/MyDrive/V4/{self.session}/{self.session}_pca.npy')
        return pca_features

    def forward(self,input_images):

        X = self.features(input_images)
        return X

            
        
