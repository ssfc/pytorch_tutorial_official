# https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
# We use transforms to perform some manipulation of the data and make it suitable for training. 

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # transform to modify the features
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))  # target_transform to modify the labels
)

###########################################################################################################################
# 1: ToTensor()
# ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image’s pixel intensity values in the range [0., 1.]

###########################################################################################################################
# 2: Lambda Transforms
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))  # turn the integer into a one-hot encoded tensor; 


















