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
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()


    def forward(self, x):  # Every nn.Module subclass implements the operations on input data in the forward method.
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

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

# (2) nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)  # The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
hidden1 = layer1(flat_image)
print(hidden1.size())

# (3) nn.ReLU
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)  # They are applied after linear transformations to introduce nonlinearity; 
print(f"After ReLU: {hidden1}")

# (4) nn.Sequential
# The data is passed through all the modules in the same order as defined; 
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
print("logits size: ", logits.size())
print("logits: ", logits)

# (5) nn.Softmax
# The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class; 
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print("pred_probab: ", pred_probab)

######################################################################################################################################
# 4: Model parameters; 
print("Model structure: ", model, "\n\n")

# makes all parameters accessible using your model’s parameters() or named_parameters() methods; 
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")














