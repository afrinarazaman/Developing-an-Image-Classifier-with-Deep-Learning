# Imports here
import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim

def build_model(model_name, hidden_units):
    
    if model_name == 'vgg-13':
       model = models.vgg13(pretrained=True)
       
    elif model_name == 'vgg-16':
       model = models.vgg16(pretrained=True)
       
    elif model_name == 'vgg-19':
       model = models.vgg19(pretrained=True)
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    input = model.classifier[0].in_features 
      
    classifier = nn.Sequential(OrderedDict([
        
        ('fc1', nn.Linear(input, hidden_units, bias=True)),
        ('relu1', nn.ReLU(inplace=True)),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    return model_name, model
    