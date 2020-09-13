import csv
import glob
import math
import os

import torch
from astropy.io import fits
from six.moves import urllib

import torch
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random
import itertools
import numpy as np


def load_GG2_images2(images):
    """
    Normalizes images and upscales them
    """


    images = [fits.open(file, memmap=False)[0].data for file in images]
    images = [torch.from_numpy(x.byteswap().newbyteorder()) for x in images]

    #Normailze
    normalize = [3.5239e+10, 1.5327e+09, 1.8903e+09, 1.2963e+09] #normalizations for 4 images
    images = [x.mul(n) for x, n in zip(images, normalize)]


    #Upscale 66*66 Images

    othersv = torch.stack(images[1:])
    upsample = torch.nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True)
    others_upsampled = torch.squeeze(upsample(othersv.unsqueeze(0)))
    
    return torch.cat((images[0].unsqueeze(0), others_upsampled),dim=0)

def load_GG2_imagesTransfer(images):
    """
    Normalizes images and does NOT upscales them and returns only visible part
    """


    images = [fits.open(file, memmap=False)[0].data for file in images]
    images = [torch.from_numpy(x.byteswap().newbyteorder()) for x in images]

    #Normailze
    normalize = [3.5239e+10, 1.5327e+09, 1.8903e+09, 1.2963e+09] #normalizations for 4 images
    images = [x.mul(n) for x, n in zip(images, normalize)]


    visible = torch.stack(images[1:])    
    return visible

def label_tansform_basic(labels):
    return (int(labels['n_sources']))*2.0 -1.0

def load_GG2_images(images):
    """
    Normalizes images and upscales them
    """


    images = [fits.open(file, memmap=False)[0].data for file in images]
    images = [torch.from_numpy(x.byteswap().newbyteorder()) for x in images]

    #Normailze
    normalize = [3.5239e+10, 1.5327e+09, 1.8903e+09, 1.2963e+09] #normalizations for 4 images
    images = [x.mul(n) for x, n in zip(images, normalize)]

    return images[0].unsqueeze(0), torch.stack(images[1:])

class GG2(torch.utils.data.Dataset):
    url_train = 'http://metcalf1.difa.unibo.it/DATA3/datapack2.0train.tar.gz'
    url_train_log = 'http://metcalf1.difa.unibo.it/DATA3/image_catalog2.0train.csv'


    def __init__(self, root, data_augmentation=False, transform=load_GG2_imagesTransfer,target_transform =  label_tansform_basic):
    #Upscale
        """
        Initializes the dataset with images and labels using the root path given.
        The images are transformed using the transfomation given in the second argument.
        """
        self.root = os.path.expanduser(root)
        self.files = None
        self.data = None
        #self.tar = None
        self.download()
        self.transform = transform
        self.target_transform= target_transform
        self.data_augmentation=data_augmentation

    def __getitem__(self, index):
        images = self.files[index]
        #files = [self.tar.extractfile(self.tar.getmember(x)) for x in images]
        
        ID = int(images[0].split('-')[-1].split('.')[0])

        if self.transform:
            #files = self.transform(files)
            images = self.transform(images)
            if self.data_augmentation:
                transform = {1: flip_horizontal , 2: flip_vertical, 3: flip_on_diagonal_that_goes_down, 4: flip_on_diagonal_that_goes_up, 5: identity, 6: rotate_by_90_deg, 7: rotate_by_180_deg, 8: rotate_by_270_deg}
                num_possible_transf=len(transform)
                which_transformation=np.random.randint(1, high=num_possible_transf+1)
                images= transform[which_transformation](images)

        labels = self.data[ID]
        if self.target_transform:
            labels = self.target_transform(labels)
        

        return images, labels 


    def __len__(self):
        return len(self.files)

    def get_labels(self):
        return torch.tensor( [self.data[i]['n_sources']*2.0-1.0 for i in self.data ])

    def download(self):
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        log_path = os.path.join(self.root, "train.csv")
        if not os.path.isfile(log_path):
            print("Download log...", flush=True)
            data = urllib.request.urlopen(self.url_train_log)
            with open(log_path, 'wb') as f:
                f.write(data.read())

        keys = [
            '',              'ID',           'x_crit',            'y_crit',
            'source_ID',     'z_source',     'z_lens',            'mag_source',
            'ein_area',      'n_crit',       'r_source',          'crit_area',
            'n_pix_source',  'source_flux',  'n_pix_lens',        'lens_flux',
            'n_source_im',   'mag_eff',      'sb_contrast',       'color_diff',
            'n_gal_3',       'n_gal_5',      'n_gal_10',          'halo_mass',
            'star_mass',     'mag_lens',     'n_sources'
        ]
        assert len(keys) == 27
        with open(log_path, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = [x for x in reader if len(x) == 27 and not 'ID' in x]
            data = [{k: float(x) if x else math.nan for k, x in zip(keys, xs)} for xs in data]
            self.data = {x['ID']: x for x in data}

        gz_path = os.path.join(self.root, "datapack2.0train.tar.gz")
        if not os.path.isfile(gz_path):
            print("Download...", flush=True)
            data = urllib.request.urlopen(self.url_train)
            with open(gz_path, 'wb') as f:
                f.write(data.read())

        tar_path = os.path.join(self.root, "datapack2.0train.tar")
        if not os.path.isfile(tar_path):
            print("Decompress...", flush=True)
            import gzip
            import shutil
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tar_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        dir_path = os.path.join(self.root, "datapack2.0train")
        if not os.path.isdir(dir_path):
            print("Extract...", flush=True)
            import tarfile
            tar = tarfile.open(tar_path)
            tar.extractall(dir_path)
            tar.close()
        
        # print("Open tar...", flush=True)
        # import tarfile
        # self.tar = tarfile.open(tar_path)

        self.files = list(zip(*(
            sorted(glob.glob(os.path.join(dir_path, "Public/{}/*.fits".format(band))))
            for band in ("EUC_VIS", "EUC_J", "EUC_Y", "EUC_H")
        )))
        assert all(len({x.split('-')[-1] for x in fs}) == 1 for fs in self.files)



def inf_shuffle(xs):
    while xs:
        random.shuffle(xs)
        for x in xs:
            yield x

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)

class BalancedBatchSampler2(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        from collections import defaultdict
        if hasattr(dataset, 'dataset'):
            transform = dataset.dataset.transform
            dataset.dataset.transform = None # trick to avoid useless computations
            indices = defaultdict(list)
            for subset_index, full_data_index in enumerate(dataset.indices):
                _, label = dataset.dataset[full_data_index]
                indices[label].append(subset_index) 
            dataset.dataset.transform = transform
        else:
            transform = dataset.transform
            dataset.transform = None  # trick to avoid useless computations
            indices = defaultdict(list)
            for i in range(0, len(dataset)):
                _, label = dataset[i]
                indices[label].append(i)
            dataset.transform = transform     

        self.indices = list(indices.values())
        self.n = max(len(ids) for ids in self.indices) * len(self.indices)


    def __iter__(self):
        m = 0
        for xs in zip(*(inf_shuffle(xs) for xs in self.indices)):
            for i in xs:  # yield one index of each label
                yield i
                m += 1
                if m >= self.n:
                    return

    def __len__(self):
        return self.n

def random_splitY(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths)).tolist()
    return indices, [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in
     zip(itertools.accumulate(lengths), lengths)]


def accuracy(net, loader,device="cpu"):
    r"""
    Calculates a net's accuracy for a given testset using its dataloader.

    Arguments:
        loader (Dataloader): Dataloader for the testset
        net (pytorch nn): neuralnet used for predictions
    """
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            predicted = net(images)
            #print(predicted.squeeze())
            predicted = torch.sign(predicted)
            #print(predicted.squeeze())
            #print(labels.squeeze())
            total += labels.size(0)
            correct += (predicted.squeeze() == labels.squeeze()).long().sum().item()
    return correct/total



#----------------------transforms for data augmentation-----------------------
def flip_horizontal(tensor):
    return tensor.flip(1)
def flip_vertical(tensor):
    return tensor.flip(2)
def rotate_by_90_deg(tensor):
    return tensor.transpose(1,2).flip(1)
def rotate_by_270_deg(tensor):
    return tensor.transpose(1,2).flip(2)
def rotate_by_180_deg(tensor):
    return rotate_by_90_deg(rotate_by_90_deg(tensor))
def identity(tensor):
    return tensor
def flip_on_diagonal_that_goes_down(tensor):
    return tensor.transpose(1,2)
def flip_on_diagonal_that_goes_up(tensor):
    return rotate_by_270_deg(flip_on_diagonal_that_goes_down(rotate_by_90_deg(tensor)))
#-----------------------------------------------------


#----------------------Functions for main-----------------------

def MakingDatasets(datapath,transfer_learning, PathDataset,data_augmentation,batch_sizev,test_batch_size,proportion_traindata):
    r"""
    Imports test and training datasets and downloads and creates them if necessary.
    Arguments:
        datapath (string): path to dataset
        transfer_learning (boolean): Whether to use transfer learning with freezing or not
        PathDataset (string): path for creating or loading the dataset
        data_augmentation (boolean): whether or not to use data augmentation
        batch_sizev (int): batch size for the testing dataloader
        test_batch_size (int): batch size for the test dataloader
        proportion_traindata (float): proportion of training data in the whole dataset
    """
    if transfer_learning:
        transform=load_GG2_imagesTransfer
    else:
        transform=load_GG2_images2

    import os
    if os.path.isfile(PathDataset):
        if os.stat(PathDataset).st_size > 0:
            import pickle
            with open(PathDataset, 'rb') as pickle_file:
                [full_dataset,trainset,testset] = pickle.load(pickle_file)
            full_dataset.transform=transform
            trainset.transform=transform
            testset.transform=transform
            print("Loading datasets...")
    else: 
        full_dataset = GG2(datapath,data_augmentation=False,transform=transform)

        # To split the full_dataset
        train_size = int(proportion_traindata * len(full_dataset))
        test_size = len(full_dataset) - train_size
        indices, sets = random_splitY(full_dataset, [train_size, test_size])
        [trainset, testset]=sets

        import pickle
        with open(PathDataset, 'wb') as pickle_file:
            pickle.dump([full_dataset,trainset,testset],pickle_file)
        print("Creating and pickling datasets...")

    # Data augmentation

    if data_augmentation:
        full_dataset.data_augmentation=True
        trainset.data_augmentation=True
        testset.data_augmentation=True

    print(len(trainset))

    # Dataloaders

    batch_sizev=8
    test_batch_size = 8

    samplerv= BalancedBatchSampler2(trainset)
    samplertest = BalancedBatchSampler2(testset)

    trainloader = torch.utils.data.DataLoader(trainset, sampler=samplerv, shuffle=False, batch_size= batch_sizev)
    testloader = torch.utils.data.DataLoader(testset, sampler=None, shuffle =True, batch_size= test_batch_size)
    ROCloader = torch.utils.data.DataLoader(testset,batch_size=1)

    return trainloader, testloader, ROCloader

# Replace all batch normalization layers by Instance
def convert_batch_to_instance(model):
    r"""
    Replace all batch normalization layers by Instance

    Arguments:
        Model : The model to which this is applied
    """

    import torch.nn as nn
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features= child.num_features
            setattr(model, child_name, nn.InstanceNorm2d(num_features=num_features))
        else:
            convert_batch_to_instance(child)

# For initializing the batch normalization layers
def init_batchnorm(model):
    r"""
    Reinitialises all batch normalization layers

    Arguments:
        Model : The model to which this is applied
    """

    import torch.nn as nn
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features= child.num_features
            setattr(model, child_name, nn.BatchNorm2d(num_features=num_features))
        else:
            convert_batch_to_instance(child)

def output(testloader,device,net):
    r"""
    Plots ROC, calculates AUROC, outputs all predictions and labels for the testset to a local csv file

    Arguments:
        testloader : The dataloader for the testset
    """
    predictions = []
    labels = []
    for k, testset_partial in enumerate(testloader):
        if k <10000: #change this theshhold to just estimate the auc using a sample of the testing data
            testset_partial_I , testset_partial_labels = testset_partial[0].to(device), testset_partial[1].to(device)
            predictions += [p.item() for p in net(testset_partial_I) ]
            labels += testset_partial_labels.tolist()
        else: break
        if k%1000==0 and not k== 0:
            print(k)
    from sklearn import metrics

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

    # importing the required module 
    import matplotlib.pyplot as plt 
    
    # x axis and y axis values 
    x ,y = fpr, tpr

    # plotting the points  
    plt.plot(x, y,marker='x') 
    plt.plot(x, x,marker='x')
    
    # naming the x axis 
    plt.xlabel('False Positive Rate') 
    # naming the y axis 
    plt.ylabel('True Positive Rate') 
    
    # giving a title to my graph 
    plt.title('Reciever operating characteristic curve') 
    
    # function to show the plot 
    plt.show()

    print("Calculating AUROC...")

    auc = metrics.roc_auc_score(labels, predictions)
    print("Test AUROC: %5f"%auc)

    print("Outputting predictions and labels for testset...")
    np.savetxt("PredictionsAndLabels.csv", [predictions,labels], delimiter=",")

def ImportNN(simple, transfer_learning):
    r"""
    Imports efficientnet from github

    Arguments:
        simple (boolean) : Whether to import the simple NN or not
        transfer_learning (boolena): Whether to freeze first layer or not
    """
    if simple:
        net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_mobilenetv3_small_minimal_100',
        pretrained=False)

        # Change First and Last Layer
        if not transfer_learning:
            net.conv_stem = torch.nn.Conv2d(4,16,kernel_size=(2,2),bias=False)
        net.classifier = torch.nn.Linear(1024, 1)
    else: 
        net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0',
        pretrained=True)

        # Change First and Last Layer
        if not transfer_learning:
            net.conv_stem = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier = torch.nn.Linear(1280, 1)
    return net


def train_load(device, PathModel, net, use_saved_model,save_trained_model, lrv, momentumv,
 transfer_learning, train_or_not, trainloader, printevery, epochs):
    r"""
    Function to load and train a neural net.

    """
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    import torch


    print("Learning rate= "+str(lrv))


    #Option to use a saved model parameters
    if use_saved_model:
        import os
        if os.path.isfile(PathModel):
            if os.stat(PathModel).st_size > 0:
                net.load_state_dict(torch.load(PathModel,map_location=torch.device(device)))
                print("Loading model...")
            else: 
                print("Empty file...")
            print("Using saved model...")


    #Training starts

    criterion = nn.SoftMarginLoss()

    if not transfer_learning:
        optimizer = optim.SGD(net.parameters(), lr=lrv, momentum=momentumv)
    else:
        optimizer = optim.SGD(net.classifier.parameters(), lr=lrv, momentum=momentumv)
        for param in net.parameters():
            param.requires_grad = False
        for param in net.classifier.parameters():
            param.requires_grad = True
        
        
    # Decay LR by a factor of 0.1 every 7 epochs
    from torch.optim import lr_scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    net.train()

    if train_or_not:
        print("Starting training...")
        train_auc_list = np.array([0])
        test_auc_list = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            exp_lr_scheduler.step()
            print("Starting epoch %d"%(epoch+1))
            print("Learning rate= "+str(lrv))
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = torch.unsqueeze(labels, dim =1)
                labels = labels.float()
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % printevery == printevery-1:    # print every n mini-batches
                    print('[%5d, %5d] loss: %.6f ' %
                            (epoch+1, i + 1, running_loss/printevery) )
                    running_loss = 0.0
            
            # save predictions and labels for ROC curve calculation
            print("Calculating AUROC...")

            # AUC for ROC curve, stop if test AUROC decreases significantly   
            if use_saved_model == 'Model1':      
                net.eval()
            elif use_saved_model == 'Model2':
                net.train()

            from sklearn import metrics
            predictions = []
            labels = []
            with torch.no_grad():
                for k, testset_partial in enumerate(testloader):
                    if k <100000:
                        testset_partial_I , testset_partial_labels = testset_partial[0].to(device), testset_partial[1].to(device)
                        predictions += [p.item() for p in net(testset_partial_I) ]
                        labels += testset_partial_labels.tolist()
                    else: break

                auc = metrics.roc_auc_score(labels, predictions)
                test_auc_list = np.concatenate((train_auc_list, np.array([auc])))
                if auc < np.max(test_auc_list)-0.04:
                    break
                print("Test auc: %5f"%auc)

            net.train()
        
        print('Finished Training')
        if save_trained_model:
            import os
            if os.path.exists(PathModel):  # checking if there is a file with this name
                os.remove(PathModel)  # deleting the file
            torch.save(net.state_dict(), PathModel)
            print("Saving model...")
    return net
        
