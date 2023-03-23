import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
import os
import dir_local as ct

#Seismic image parameters
data_mean = 0.4746
data_std = 0.1526

# amplitude for the noise augmentation
augm_sigma = 0.08

def unnormalize(x):
    '''go from normaized data x back to the original range'''
    return x * data_std + data_mean

def normalization(x):
    return (x - data_mean) / data_std

def noise(x):
    return (x + augm_sigma * torch.randn_like(x))

class SeismicImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, label_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = label_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx, 0] + '.png'
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        label = 1            #self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    


# Custom Dataset Partition
batch_size = 128
transform = T.Compose([T.Resize((32 , 32)),
                       T.Grayscale(num_output_channels=1),
                       T.RandomHorizontalFlip(),
                       T.ToTensor()
                       #,T.Normalize(data_mean, data_std, inplace=False) 
                     ])

transformV = T.Compose([T.Resize((32 , 32)),
                       T.Grayscale(num_output_channels=1),
                       T.ToTensor()
                       #,T.Normalize(data_mean, data_std, inplace=False) 
                     ])

# Use ImageFolder to create dataset(s)
from torchvision import datasets
full_dataset = datasets.ImageFolder(root=r"E:\Luciano\_0PH\Datasets\tgs-salt\test\images", # target folder of images
                                  transform=transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

#test_data = datasets.ImageFolder(root=r"E:\Luciano\_0PH\Datasets\tgs-salt\train\images", 
#                                 transform=transformV)

#full_dataset = SeismicImageDataset(ct.TEST_NOT_NULLCSV,ct.TRAIN_IMAGE_DIR, transform)

dataset_size = len(full_dataset)
validation_split = .15
random_seed= 42
split = int(np.floor(dataset_size * validation_split))


# new splitted datasets 
# https://bit.ly/3kyZooA   How to split a custom dataset into training and test datasets
valid_split = split
train_split = dataset_size - split
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_split, valid_split])

# Sample a fixed batch of 200 validation examples (split=validationSamples)
#val_x, val_l = zip(*list(valid_dataset[i] for i in range(split)))
#val_x = torch.stack(val_x, 0).cuda()
#val_l = torch.LongTensor(val_l).cuda()

# Add the noise-augmentation to the training data only:
#train_dataset.transform = T.Compose([train_dataset.transform, T.Lambda(noise)])   

# create batches loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Creating data indices for training and validation splits:
#indices = list(range(dataset_size))
#train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
# Exclude the validation batch from the training data
#train_sampler = SubsetRandomSampler(train_indices)
#valid_sampler = SubsetRandomSampler(val_indices)

#train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
#valid_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_sampler)