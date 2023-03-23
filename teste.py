import dir_local as dl
import torch
import torchvision
import os
import random

root = dl.root
folder = dl.folder + str(int(1000*random.random()))

print (root, "   " , folder)
os.makedirs(folder, exist_ok=True)
