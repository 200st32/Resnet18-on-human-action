import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import opendatasets as od
from tqdm import tqdm
import numpy as np
import sys
from pynvml import *
import timeit

def getdata(batch_size):

    isExist = os.path.exists("./human-action-recognition-dataset")
    if isExist==False:
        dataset = 'https://www.kaggle.com/datasets/shashankrapolu/human-action-recognition-dataset/data'
        od.download(dataset)
    else:
        print("dataset exist")

    # Load the datasets
    data_dir =  "./human-action-recognition-dataset/Structured/"
    
    data_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]) 

    # do data augmentation to get more train data
    data_transforms_train = transforms.Compose([
        transforms.RandomRotation(20),  # Randomly rotate the image within a range of (-20, 20) degrees
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with 50% probability
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop the image and resize it
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation, and hue
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.5),  # Randomly apply affine transformations with translation
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.5),  # Randomly apply perspective transformations
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Split out val dataset from train dataset
    train_dataset = datasets.ImageFolder(data_dir+"train/", transform=data_transforms_train)
    n = len(train_dataset)
    n_val = int(0.1 * n)
    val_dataset = torch.utils.data.Subset(train_dataset, range(n_val))
    test_dataset = datasets.ImageFolder(data_dir+"test/", transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    '''
    class_names = train_loader.dataset.classes
    for i in class_names:
        print(i)
    '''
    print("data load successfully!")
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, optimizer, loss_function, device):
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))
    batch_loss=[]
    nvmlInit()
    start = timeit.default_timer()
    for batch in pbar:
        if batch is None:
            continue
            
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
        pbar.set_description(f"Train Loss: {loss.item():.4f}")
        
    batch_loss = np.array(batch_loss)
    stop = timeit.default_timer()
    print(f'Train Epoch Time: {stop - start:.4f} sec') 
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'Train Epoch GPU memory used: {info.used/1000000:.4f} MB')
    return batch_loss.mean()

def val_model(model, val_loader, loss_function, device):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(val_loader, total=len(val_loader))
        batch_loss=[]
        for batch in pbar:
            if batch is None:
                continue

            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            #calculate loss
            loss = loss_function(outputs, labels)
            batch_loss.append(loss.item())
            pbar.set_description(f"Validation Loss: {loss.item():.4f}")
        batch_loss = np.array(batch_loss)
        return batch_loss.mean()

def test_model(model, test_loader, loss_function, device):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader))
        batch_loss=[]
        test_running_correct = 0
        for batch in pbar:
            if batch is None:
                continue

            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            #calculate loss
            loss = loss_function(outputs, labels)
            batch_loss.append(loss.item())
            #calculate accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()
            pbar.set_description(f"Test Loss: {loss.item():.4f}")
        
        accuracy = 100.0 * (test_running_correct / len(test_loader.dataset))
        batch_loss = np.array(batch_loss)
        return batch_loss.mean(), accuracy

if __name__ == '__main__':

    getdata(16)
