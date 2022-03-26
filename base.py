# truncated model
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
class CustomResNet(nn.Module):
    def __init__(self, model,best_layer,layer_list):
        super(CustomResNet, self).__init__()
        self.model = model
        self.best_layer = best_layer
        self.layer_list = layer_list
        temp_index=layer_list.index(best_layer)
        # cut the model, only keep layers before the best layer
        rest_layers=layer_list[:temp_index+1]
        
        modules=[]
        modules.append(model.conv1)
        modules.append(model.bn1)
        modules.append(model.relu)

        for layer in rest_layers:          
            exec(f'modules.append(model.{layer})')
        
        
        self.part=nn.Sequential(*modules)

    def forward(self, x):
        x = self.part(x)
        return x