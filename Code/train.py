

import os
import time
import numpy as np
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from six.moves import urllib
from torchvision import datasets, transforms
#from torchsummary import summary
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision.transforms.functional as TF
from PIL import Image
import sys
import json
#sys.path.insert(1, '/Code') #https://stackoverflow.com/questions/4383571/importing-files-from-different-folder 
from Util import * #Util as Util #Util.__name__

print(torch.cuda.is_available())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
pd.set_option("display.width",150)
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180


import argparse
parser = argparse.ArgumentParser(description="Train Classifier")
parser.add_argument("data_dir", type=str, help='directory of data folders')
parser.add_argument("--arch", type=str, default = "vgg13", help='torchvision model name')
parser.add_argument("--hidden_units", default = 500, type=int, help='hidden layer units')
parser.add_argument("--gpu", type=bool, default = True, help='use GPU if True')
parser.add_argument("--learning_rate", default = 0.01, type=float, help='learning rate')
parser.add_argument("--epochs", type=int, default = 1, help='number of loops for training')
parser.add_argument("--save_dir", type=str, default = "checkpoint_v0.pt", help='checkpoint saving directory')
args = parser.parse_args()


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

def freeze_parameters(root, freeze=True):
    [param.requires_grad_(not freeze) for param in root.parameters()]

def train (model, optimizers, epochs=10, lr_scheduler=None,
          dataloaders=None, state_dict=None,
          checkpoint_path="checkpoint_v0.pt", accuracy_target=None,
          show_graphs=True):

    if state_dict == None:
        state_dict = {
            'elapsed_time': 0,
            'trace_log': [],
            'trace_train_loss': [],
            'trace_train_lr': [],
            'valid_loss_min': np.Inf,
            'trace_valid_loss': [],
            'trace_accuracy': [],
            'epochs_trained': 0}
        state_dict['trace_log'].append('train_v1')

    train_losses , valid_losses=[],[]
    
    for epoch in range(1,epochs+1):
        try:
            lr_scheduler.step() # if instance of _LRScheduler
        except TypeError:
            try:
                if lr_scheduler.min_lrs[0] == lr_scheduler.optimizer.param_groups[0]['lr']:
                    break
                lr_scheduler.step(valid_loss) # if instance of ReduceLROnPlateau
            except NameError: # valid_loss is not defined yet
                lr_scheduler.step(np.Inf)
        except:
            pass # do nothing

        epoch_start_time = time.time()

        #=====================Train===========================#
        tot_train_loss=0
        model.train()
        optimizer=optimizers[0]
        for images, labels in dataloaders['train']: #pass

            images, labels = images.to(device), labels.to(device)
            # [opt.zero_grad() for opt in optimizers]
            optimizer.zero_grad()
            # Pass train batch through model feed-forward
            logp=model(images)
            # Calculate loss for this train batch
            batch_loss = criterion(logp, labels)
            # Do the backpropagation
            batch_loss.backward()
            # Optimize parameters    #[opt.step() for opt in optimizers]
            optimizer.step()
            # Track train loss
            tot_train_loss += batch_loss.item()#*len(images)

        # Track how many epochs has already run
        state_dict['elapsed_time'] += time.time()-epoch_start_time
        state_dict['epochs_trained'] += 1
        print("train step is done")
        
        #==========================Validate===============================#
        tot_valid_loss = 0
        accuracy = 0
        top_class_graph = []
        labels_graph = []
        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():
            for images, labels in dataloaders['valid']: #pass
                labels_graph.extend( labels )
                # Move tensors to device
                images, labels = images.to(device), labels.to(device)
                # Get predictions for this validation batch
                logp = model(images)
                # Calculate loss for this validation batch
                batch_loss = criterion(logp, labels)
                # Track validation loss
                tot_valid_loss += batch_loss.item()#*len(images)
                # Calculate accuracy
                probabilities = torch.exp(logp)
                top_p, top_class = probabilities.topk(1, dim=1)
                top_class_graph.extend( top_class.view(-1).to('cpu').numpy() )
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()#*len(images) 
                # This accuracy will be divided with the number of batches - as shown below.  
                # Or more precisely, accuracy can be calcualted as the commented-out method,*len(images) first, then /len(dataloaders['train']).dataset
        
        #========================Print outcomes=======================#     
        # calculate average losses
        train_losses = tot_train_loss/len(dataloaders['train'])#.dataset)
        valid_losses = tot_valid_loss/len(dataloaders['valid'])#.dataset)
        accuracy = accuracy/len(dataloaders['valid'])#.dataset)

        state_dict['trace_train_loss'].append(train_losses)
        try:
            state_dict['trace_train_lr'].append(lr_scheduler.get_lr()[0])
        except:
            state_dict['trace_train_lr'].append(optimizers[0].state_dict()['param_groups'][0]['lr'])
        state_dict['trace_valid_loss'].append(valid_losses)
        state_dict['trace_accuracy'].append(accuracy)

        # print training/validation statistics 
        log = 'Epoch: {}: \
               lr: {:.8f}\t\
               Training Loss: {:.6f}\t\
               Validation Loss: {:.6f}\t\
               Validation accuracy: {:.2f}%\t\
               Elapsed time: {:.2f}'.format(
                    state_dict['epochs_trained'],
                    state_dict['trace_train_lr'][-1],
                    train_losses,
                    valid_losses,
                    accuracy*100,
                    state_dict['elapsed_time']
                    )
        state_dict['trace_log'].append(log)
        print(log)

        # save model if validation loss has decreased
        if valid_losses <= state_dict['valid_loss_min']:
            print('Validation loss decreased: \
                  ({:.6f} --> {:.6f}).   Saving model ...'
                  .format(state_dict['valid_loss_min'],valid_losses))

            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizers[0].state_dict(),
                          'training_state_dict': state_dict}

            torch.save(checkpoint, checkpoint_path)
            state_dict['valid_loss_min'] = valid_losses

        # stop training loop if accuracy_target has been reached
        if accuracy_target and state_dict['trace_accuracy'][-1] >= accuracy_target:
            break

    return state_dict


def save_checkpoint(checkpoint_path='checkpoint.pt',model_name="vgg13"):
    #model.to('cpu')
    checkpoint = {'model': model_name,
                  'finalclass': finalclass,
                  'set_classifer_hiddensize': set_classifer_hiddensize,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'cat_label_to_name': cat_label_to_name,
                  'optimizer_state_dict': optimizers[0].state_dict()}
    torch.save(checkpoint, checkpoint_path)
    


if __name__ == "__main__":
    # args
    data_dir=args.data_dir
    model_name=args.arch
    set_classifer_hiddensize=args.hidden_units
    gpu = args.gpu
    lr_rate = args.learning_rate
    epochs=args.epochs
    save_dir=args.save_dir
    
    
    # Set up loader objects
    data_transforms = data_transforms()
    image_datasets = image_datasets(data_dir, data_transforms)
    dataloaders = dataloaders(image_datasets)

    class_to_idx = image_datasets['train'].class_to_idx  
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    cat_label_to_name = {}
    for cat, label in class_to_idx.items():
        name = cat_to_name.get(cat)
        cat_label_to_name[label] = name   
    finalclass=102
    model = customize_classifer (model_name = model_name,
                                 set_classifer_hiddensize=set_classifer_hiddensize,
                                 finalclass=finalclass)
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on: {str(device).upper()}')
    model.to(device)
    criterion = nn.NLLLoss() 
    
    freeze_parameters(model)
    try:
        freeze_parameters(model.fc, False)
        fc_optimizer = optim.Adagrad(model.fc.parameters(), lr=lr_rate, weight_decay=0.001)
    except AttributeError:
        freeze_parameters(model.classifier, False)
        fc_optimizer = optim.Adagrad(model.classifier.parameters(), lr=lr_rate, weight_decay=0.001)

    optimizers = [fc_optimizer]
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min',
                                                   factor=0.1, patience=5,
                                                   threshold=0.01, min_lr=0.00001)
    checkpoint_path = "checkpoint_temp.pt"
    
    state_dict = train(model, optimizers, epochs=epochs, lr_scheduler=lr_scheduler,
                       state_dict=None, accuracy_target=None,dataloaders=dataloaders,
                       checkpoint_path=checkpoint_path)
 
    save_checkpoint(checkpoint_path=save_dir,model_name=model_name)


