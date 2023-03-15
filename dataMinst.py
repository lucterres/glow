import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.datasets

batch_size = 256
data_mean = 0.128
data_std = 0.305

# amplitude for the noise augmentation
augm_sigma = 0.08
data_dir = 'E:\Luciano\_0PH\Datasets'  #'mnist_data' E:\Luciano\_0PH\Datasets\MNIST

def unnormalize(x):
    '''go from normaized data x back to the original range'''
    return x * data_std + data_mean

def normalization(x):
    return (x - data_mean) / data_std

def noise(x):
    return (x + augm_sigma * torch.randn_like(x))

transf =  T.Compose([T.ToTensor()
                     #, T.Normalize (data_mean, data_std, inplace=False)
                     ])

train_data = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transf)
test_data  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transf)

# Sample a fixed batch of 1024 validation examples
val_x, val_l = zip(*list(train_data[i] for i in range(1024)))
val_x = torch.stack(val_x, 0).cuda()
val_l = torch.LongTensor(val_l).cuda()

# Exclude the validation batch from the training data
train_data.data = train_data.data[1024:]
train_data.targets = train_data.targets[1024:]
# Add the noise-augmentation to the (non-validation) training data:
train_data.transform = T.Compose([train_data.transform, T.Lambda(noise)])    

train_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader   = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)