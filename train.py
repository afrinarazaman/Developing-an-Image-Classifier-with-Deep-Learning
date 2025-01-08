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
from utils import load_data
from model import build_model


def train_model(data_dir, arch, learning_rate, hidden_units, epochs, save_dir, gpu):
    
    trainloader, validloader, train_datasets = load_data(data_dir)
    
    model_name, model = build_model(arch, hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)
    
    print_every = 20
    for epoch in range(epochs):
        training_loss = 0
        validation_loss = 0
        accuracy = 0
        step = 0
        
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            step += 1
            optimizer.zero_grad()

            logits = model.forward(images)
            loss = criterion(logits, labels)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            if step % print_every == 0 or step == 1 or step == len(trainloader):
                print(f"Epoch: {epoch+1}/{epochs} Batch/Complete: {(step*100)/len(trainloader):.2f}%")
        
        
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                logits = model.forward(images)
                loss = criterion(logits, labels)
                validation_loss += loss.item()

                ps = torch.exp(logits)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.float()).item()
        
        training_loss /= len(trainloader)
        validation_loss /= len(validloader)
        accuracy /= len(validloader)

        print(f"Epoch {epoch+1} out of {epochs}... "
            f"Training loss: {training_loss:.3f} "
            f"Validation loss: {validation_loss:.3f} "
            f"Accuracy: {accuracy*100:.2f}%")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint = {
                'model': model_name,
                'model_classifier': model.classifier,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epochs': epoch+1,
                'class_to_idx': train_datasets.class_to_idx
            }
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a NN')
    parser.add_argument('data_dir', help="Path to the dataset")
    parser.add_argument('--save_dir', default='', help='Path to save the checkpoint')
    parser.add_argument('--arch', default='vgg-19', choices=['vgg-13', 'vgg-16', 'vgg-19'], help='Model architechture')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of Hidden Units')
    parser.add_argument('--epochs', type=int, default=15, help='Number of Epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, arch=args.arch, learning_rate=args.learning_rate, hidden_units=args.hidden_units, epochs=args.epochs, save_dir=args.save_dir, gpu=args.gpu)
    
  