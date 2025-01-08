# Imports here
import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
import random
import os
import argparse
from utils import load_data, process_image
from model import build_model
import json

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=torch.device('cpu') if device.type == 'cpu' else None)
    
    if checkpoint['model'] == "vgg-13":
        model = models.vgg13(pretrained=True)
        
    elif checkpoint['model'] == "vgg-16":
        model = models.vgg16(pretrained=True)
    
    elif checkpoint['model'] == "vgg-19":
        model = models.vgg19(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['model_classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    print("Done")
    
    return model, optimizer, checkpoint


def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model.forward(image)
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(topk, dim=1)
        
    idx_to_flower = {value: cat_to_name[key] for key, value in model.class_to_idx.items()}
    predicted_flowers = [idx_to_flower[i] for i in top_class.tolist()[0]]
    
    return top_p.tolist()[0], predicted_flowers


def predictions(args):
    
    device = torch.device('cuda' and args.gpu if torch.cuda.is_available() else 'cpu')
    model, optimizer, checkpoint = load_checkpoint(args.model_filepath, device=device)
    model = model.to(device)
    
    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)
        
    top_p, top_class = predict(args.image_path, model, args.top_k, device, cat_to_name)
    
    for i in range(args.top_k):
        print(f"{i:<3} {top_class[i]:<30} probability: {top_p[i]*100} %")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")

    parser.add_argument(dest='image_path', help="This is a image file path")
    parser.add_argument(dest='model_filepath', help="This is a checkpoint file path", default='image_classification_vgg19_checkpoint.pth')

    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', help="This is a json file path that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="This is the number of most likely classes to return, default is 5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')
    
    args = parser.parse_args()

    predictions(args)
    
    
