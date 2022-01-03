
import os
import time
import numpy as np
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from six.moves import urllib
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision.transforms.functional as TF
from PIL import Image
import json
import sys
#sys.path.insert(1, '/Code')
#import Code.Util as Util
#import Code.train as train
from Util import *
#from train import *

import argparse

parser = argparse.ArgumentParser(description="Train Classifier")
parser.add_argument("image_dir", type=str, default = "flowers/test", help='directory of test folders')
parser.add_argument("checkpoint_dir", type=str, default = "checkpoint_v0.pt", help='directory of saved model')
parser.add_argument("--data_dir", type=str, default = "flowers", help='directory of data folders')
parser.add_argument("--gpu", type=bool, default = True, help='use GPU if True')
parser.add_argument("--top_k", default = 3, type=int, help='Return top KK most likely classes')
parser.add_argument("--hidden_units", default = 500, type=int, help='hidden layer units')
args = parser.parse_args()


def load_checkpoint(checkpoint_path='checkpoint_v0.pt'):
    checkpoint = torch.load(checkpoint_path)
    model_name_curr=checkpoint['model']
    model = customize_classifer (model_name = model_name_curr, 
                                 set_classifer_hiddensize=set_classifer_hiddensize,
                                 finalclass=finalclass)
   
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.cat_label_to_name = checkpoint['cat_label_to_name']
    
    return model


def load_model(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['training_state_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
        fc_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        pass

    return state_dict


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    with torch.no_grad():
        model.eval()
        
        image = image.view(1,3,224,224)
        image = image.to(device)
        predictions = model.forward(image)
        predictions = torch.exp(predictions)
        top_ps, top_class = predictions.topk(topk, dim=1)
    
    return top_ps, top_class



#image_dir = "flowers/test"
#gpu=True
#checkpoint_dir = 'checkpoint_v0.pt'
#data_dir="flowers"
if __name__ == "__main__":
    # args
    image_dir=args.image_dir #"flowers/test"
    checkpoint_dir = args.checkpoint_dir
    data_dir=args.data_dir
    gpu = args.gpu
    set_classifer_hiddensize=args.hidden_units
    top_k = args.top_k
    finalclass=102

    data_transforms = data_transforms()
    image_datasets = image_datasets(data_dir, data_transforms)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    class_to_idx = image_datasets['test'].class_to_idx
    
    cat_label_to_name = {}
    for cat, label in class_to_idx.items():
        name = cat_to_name.get(cat)
        cat_label_to_name[label] = name
    #print(cat_label_to_name)
    
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on: {str(device).upper()}')
  
    checkpoint_path = checkpoint_dir
    
    model = load_checkpoint(checkpoint_path)
    model.to(device)
    
    folder = random.choice(os.listdir(image_dir))
    image_file = random.choice(os.listdir(os.path.join(image_dir, folder)))
    image_path = image_dir + f'/{folder}/{image_file}'
    
    probs, classes = predict(image_path, model, topk=top_k)
    print(probs)
    print(classes)
    
    classes = classes.data.cpu()
    classes = classes.numpy().squeeze()
    classes = [cat_label_to_name[i].title() for i in classes]
    
    label = class_to_idx[str(folder)]
    title = f'{cat_label_to_name[label].title()}'
    
    print(f"Predicted top classes: {classes}")
    print(f"True class: {title}")
    