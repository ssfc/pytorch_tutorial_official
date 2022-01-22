# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

import os
import torch
from torch import nn  # The torch.nn namespace provides all the building blocks you need to build your own neural network; 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

######################################################################################################################################
# 1: Get Device for Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # We want to be able to train our model on a hardware accelerator like the GPU, if it is available
print(f'Using {device} device')

######################################################################################################################################
# 2: Define the Class
class NeuralNetwork(nn.Module):  # We define our neural network by subclassing nn.Module; 
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits





























