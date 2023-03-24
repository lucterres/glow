import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import dir_local as dl


tgsroot = dl.tgsroot

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
full_dataset = datasets.ImageFolder(root=tgsroot + "/test", # target folder of images
                                  transform=transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

#test_data = datasets.ImageFolder(root=r"E:\Luciano\_0PH\Datasets\tgs-salt\train\images", 
#                                 transform=transformV)

#full_dataset = SeismicImageDataset(ct.TEST_NOT_NULLCSV,ct.TRAIN_IMAGE_DIR, transform)

dataset_size = len(full_dataset)
validation_split = .15
random_seed= 42
split = int((dataset_size * validation_split))  #np.floor


# new splitted datasets 
# https://bit.ly/3kyZooA   How to split a custom dataset into training and test datasets
valid_split = split
train_split = dataset_size - split
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_split, valid_split])

  

# create batches loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

