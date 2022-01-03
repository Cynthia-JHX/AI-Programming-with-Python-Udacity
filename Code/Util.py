# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:34:25 2021

@author: Shafufu
"""

import os
import time
import numpy as np
import torch
import random
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from six.moves import urllib
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision.transforms.functional as TF
from PIL import Image
import json
print(torch.cuda.is_available())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
pd.set_option("display.width",150)
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
os.getcwd()
#os.chdir('C:\\Users\\Shafufu\\Desktop\\Huacheng Doc\\HL Python Learning\\Udacity\\Project_2')
#data_dir='flowers'
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir  = data_dir + '/test'
normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])




def data_transforms():
    data_transforms = {}
    data_transforms["train"] = transforms.Compose([
                                           transforms.RandomChoice([
                                               transforms.RandomRotation(180),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomVerticalFlip(p=0.5)]),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalize_mean,normalize_std)])
    data_transforms["valid"] = transforms.Compose([
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalize_mean,normalize_std)])
    data_transforms["test"]  = transforms.Compose([
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalize_mean,normalize_std)])
    return(data_transforms)


def image_datasets(data_dir, data_transforms):
    image_datasets = {}
    image_datasets["train"] = datasets.ImageFolder(data_dir + '/train', transform=data_transforms["train"])
    image_datasets["valid"] = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms["valid"])
    image_datasets["test"]  = datasets.ImageFolder(data_dir + '/test' , transform=data_transforms["test"])
    return image_datasets

def dataloaders(image_datasets):
    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, shuffle=True)
    dataloaders["valid"] = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32, shuffle=False)
    dataloaders["test"]  = torch.utils.data.DataLoader(image_datasets["test"],  batch_size=32, shuffle=False)
    return dataloaders
    
def customize_classifer (model_name = "vgg13", set_classifer_hiddensize=500,finalclass=102, dropout=0.5):
    model = getattr(models, model_name)(pretrained=True)   
    for param in model.parameters():
        param.requires_grad = False
    classifier_name, old_classifier = model._modules.popitem()
    try:
        old_classifier=old_classifier[0] # when model is vgg13
    except TypeError:
        pass
    classifier_input_size = old_classifier.in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_input_size, set_classifer_hiddensize)),
                          ('relu', nn.RReLU()),
                          ('drop',nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(set_classifer_hiddensize,finalclass)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.add_module(classifier_name, classifier)
    return model    

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    image = TF.resize(image, 256)
    upper_pixel = (image.height - 224) // 2
    left_pixel = (image.width - 224) // 2
    image = TF.crop(image, upper_pixel, left_pixel, 224, 224)
    image = TF.to_tensor(image)
    image = TF.normalize(image, normalize_mean, normalize_std)
    
    return image


def imshow(image, ax=None, title=None, titlecolor='k'):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.grid(False)
    if title:
        ax.set_title(title, color=titlecolor)
    
    return ax


#load model
def load_checkpoint(checkpoint_path='checkpoint.pt'):
    checkpoint = torch.load(checkpoint_path)

    model = customize_classifer (select_model = models.resnet152, 
                                 set_classifer_hiddensize=set_classifer_hiddensize,
                                 finalclass=finalclass)
   
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.cat_label_to_name = checkpoint['cat_label_to_name']
    
    return model


if __name__ == "__main__":
    # Test flower_names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    for i, key in enumerate(cat_to_name.keys()):
        print(key, ':', cat_to_name[key])
        if i == 5:
            break
    data_dir = 'flowers'
    # Test creation of all input files
    data_transforms = data_transforms()
    image_datasets = image_datasets(data_dir, data_transforms)
    dataloaders = dataloaders(image_datasets)
    # If three loader objects print, setup is complete
    for dataloader in dataloaders:
        print(dataloader)
    class_to_idx = image_datasets['train'].class_to_idx  
    cat_label_to_name = {}
    for cat, label in class_to_idx.items():
        name = cat_to_name.get(cat)
        cat_label_to_name[label] = name

    # Test image processing
    image_folder = 'flowers/train/'
    random_folder = random.choice(os.listdir(image_folder))
    random_file = random.choice(os.listdir(os.path.join(image_folder, random_folder)))
    image_path = os.path.join(image_folder, random_folder, random_file)
    image = Image.open(image_path)
    processed_image = process_image(image)
    display_image = imshow(processed_image)
    plt.show()








































