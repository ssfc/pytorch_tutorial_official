# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

#################################################################################################################################
# 1: Loading a Dataset
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data", # root is the path where the train/test data is stored; 
    train=True, # train specifies training or test dataset; 
    download=True, # download=True downloads the data from the internet if it’s not available at root; 
    transform=ToTensor() # transform and target_transform specify the feature and label transformations; 
)

test_data = datasets.FashionMNIST(
    root="data", # root is the path where the train/test data is stored; 
    train=False, # train specifies training or test dataset; 
    download=True, # download=True downloads the data from the internet if it’s not available at root; 
    transform=ToTensor() # transform and target_transform specify the feature and label transformations; 
)

#################################################################################################################################
# 2: Iterating and Visualizing the Dataset



























