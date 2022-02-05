# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

# SAVING AND LOADING A GENERAL CHECKPOINT IN PYTORCH

##################################################################################################################################################################
# 1: Introduction

##################################################################################################################################################################
# 2: Setup

##################################################################################################################################################################
# 3: Steps

# (1) Import necessary libraries for loading our data; 
import torch
import torch.nn as nn
import torch.optim as optim

# (2) Define and intialize the neural network; 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)






























