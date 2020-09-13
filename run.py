# %% Global parameters

import torch

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

#Global variables (default values):
transfer_learning=0
init_batchnormv =1
use_parallelization=0
simple =0
data_augmentation =0

use_saved_model ='Model1'
save_trained_model=0

train_or_not =0
epochs =20

proportion_traindata = 0.8 # the proportion of the full dataset used for training
printevery = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


momentumv=0.90
lrv=10**-2

train_batch_size=8

import argparse
parser = argparse.ArgumentParser()

# the proportion of the full dataset used for training
parser.add_argument("--ptd", type=int, default=proportion_traindata, 
    help="the proportion of the full dataset used for training") 
# training dataloader batchsize
parser.add_argument("--tbs", type=int, default=train_batch_size, help="training dataloader batchsize")
# learning rate
parser.add_argument("--lr", type=float, default=lrv,help="learning rate")
# momentum
parser.add_argument("--mom", type=float, default=momentumv,help="momentum")
# number of epochs 
parser.add_argument("--epoch", type=int, default=epochs,help="number of epochs ")
# device
parser.add_argument("--device", type=str, default=device)
# path
parser.add_argument("--root", type=str, default=dir_path, help="the working directory path" )
# model
parser.add_argument("--model", type=str, default=use_saved_model, help= "model to use")
# Whether to save model after training model
parser.add_argument("--save", type=int, default=save_trained_model, help= "whether to save model after training model")
# print every
parser.add_argument("--pevery", type=int, default=printevery, help = "how often to print training steps")
# transfer learning boolean
parser.add_argument("--transfer", type=int, default=transfer_learning,help= "whether or not to freeze layers")
# Whether or not to init the batch normalization layers
parser.add_argument("--initbatch", type=int, default=init_batchnormv, help= "whether or not to init the batch normalization layers")
# Whether or not to parallelize
parser.add_argument("--parallelize", type=int, default=use_parallelization, help= "whether or not to parallelize")
# Whether or not to use data augmentation
parser.add_argument("--dataaugmentation", type=int, default=data_augmentation, help="whether or not to use data augmentation")
# Whether or not to train the net
parser.add_argument("--train", type=int, default=train_or_not,help= "whether or not to train the net")

# Global variables reassignment: 
args = parser.parse_args()

transfer_learning=args.transfer
init_batchnormv =args.initbatch
use_parallelization=args.parallelize
simple =0
data_augmentation =args.dataaugmentation

use_saved_model = args.model
save_trained_model=args.save

train_or_not =args.train
epochs = args.epoch

proportion_traindata = args.ptd # the proportion of the full dataset used for training
printevery = args.pevery
device = args.device


momentumv=args.mom
lrv=args.lr

train_batch_size=args.tbs

if use_saved_model == "Model2":
    init_batchnormv =0
    data_augmentation =0
    use_parallelization =0
if use_saved_model == "Model1":
    use_parallelization =1



# PLEASE INSERT YOUR PATH HERE
PathModel= args.root+'/'+use_saved_model +'.modeldict'
PathDataset = args.root +'/traintestsets.pckl'
datapath = args.root+'/Data' 


# %% Import Dataset and create trainloader 
import datasetY as dataset
import importlib
from datasetY import BalancedBatchSampler, BalancedBatchSampler2, random_splitY, accuracy, load_GG2_imagesTransfer, load_GG2_images2
import itertools
import numpy as np



# Pickling datasets

from datasetY import MakingDatasets
trainloader, testloader, ROCloader = MakingDatasets(datapath= datapath,transfer_learning=transfer_learning, PathDataset=PathDataset
                                            ,data_augmentation=data_augmentation,batch_sizev=args.tbs,test_batch_size=8,
                                            proportion_traindata=proportion_traindata)
# %% Import Neural network
from datasetY import init_batchnorm, ImportNN

net = ImportNN(simple,transfer_learning)
net.to(device)

#Converts model with init_batchnorm
if not transfer_learning and init_batchnormv:
        init_batchnorm(net)

#Option to parallelize
print("There are", torch.cuda.device_count(), "GPUs!")
if use_parallelization:
    import torch.nn as nn
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
net.to(device)


# %% Train Neural network


# To calculate accuracy
from sampler import accuracy

def train_accuracy(net):
    return accuracy(net, loader= trainloader,device=device)

def test_accuracy(net):
    return accuracy(net, loader= testloader,device=device)

def ROC_accuracy(net):
    return accuracy(net, loader= ROCloader,device=device)

# Training/ loading:
from datasetY import train_load
net = train_load(device, PathModel, net, use_saved_model,save_trained_model, lrv, momentumv,
 transfer_learning, train_or_not, trainloader, printevery, epochs)      

# %% Outputs: Plots ROC, calculates AUROC, outputs all predictions and labels for the testset to a local csv file

with torch.no_grad():
    if use_saved_model == "Model2":
        net.train() 
    else:
        net.eval()
    from datasetY import output
    output(testloader,device, net)  


