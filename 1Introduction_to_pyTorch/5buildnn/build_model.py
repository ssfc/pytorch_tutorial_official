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
    def __init__(self):  # and initialize the neural network layers in __init__;
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):  # Every nn.Module subclass implements the operations on input data in the forward method.
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# We create an instance of NeuralNetwork, and move it to the device, and print its structure.
model = NeuralNetwork().to(device)
print(model)

######################################################################################################################################
# 3: Model Layers

input_image = torch.rand(3,28,28)
print(input_image.size())

# (1) nn.Flatten: 
# We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is maintained).
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())






















