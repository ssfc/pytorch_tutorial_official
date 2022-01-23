# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html

###############################################################################################################################################################
# 1: PyTorch Tensors

import torch

z = torch.zeros(5, 3)  # Above, we create a 5x3 matrix filled with zeros; 
print(z)
print(z.dtype)  # query its datatype to find out that the zeros are 32-bit floating point numbers, which is the default PyTorch; 

i = torch.ones((5, 3), dtype=torch.int16)  # What if you wanted integers instead? You can always override the default; 
print(i)

torch.manual_seed(1729)  # It’s common to initialize learning weights randomly, often with a specific seed for the PRNG for reproducibility of results; 
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # repeats values of r1 because of re-seed

# PyTorch tensors perform arithmetic operations intuitively. Tensors of similar shapes may be added, multiplied, etc. Operations with scalars are distributed over the tensor:
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # every element is multiplied by 2
print(twos)

threes = ones + twos       # additon allowed because shapes are similar
print(threes)              # tensors are added element-wise
print(threes.shape)        # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# uncomment this line to get a runtime error
# r3 = r1 + r2

# Here’s a small sample of the mathematical operations available:
r = (torch.rand(2, 2) - 0.5) * 2 # values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')
print(torch.abs(r))

# ...as are trigonometric functions:
print('\nInverse sine of r:')
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# ...and statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))

###############################################################################################################################################################
# 2: PyTorch Models
import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function


class LeNet(nn.Module):  # It inherits from torch.nn.Module - modules may be nested - in fact, even the Conv2d and Linear layer classes inherit from torch.nn.Module. 

    def __init__(self):  # A model will have an __init__() function, where it instantiates its layers, and loads any data artifacts it might need (e.g., an NLP model might load a vocabulary).
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)  # Layer C1 is a convolutional layer, meaning that it scans the input image for features it learned during training. 
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # An input is passed through the network layers and various functions to generate an output. 
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# First, we instantiate the LeNet class, and we print the net object. 
# A subclass of torch.nn.Module will report the layers it has created and their shapes and parameters. 
# This can provide a handy overview of a model if you want to get the gist of its processing.
net = LeNet()
print(net)                         # what does the object tell us about itself?

# Below that, we create a dummy input representing a 32x32 image with 1 color channel. 
# You may have noticed an extra dimension to our tensor - the batch dimension. 
input = torch.rand(1, 1, 32, 32)   # stand-in for a 32x32 black & white image
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # we don't call forward() directly
print('\nRaw output:')
print(output)
print(output.shape)

###############################################################################################################################################################
# 3: Datasets and Dataloaders

#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),  # transforms.ToTensor() converts images loaded by Pillow into PyTorch tensors. 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transforms.Normalize() adjusts the values of the tensor so that their average is zero and their standard deviation is 0.5. 


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)  # organizes the input tensors served by the Dataset into batches with the parameters you specify.

###############################################################################################################################################################
# 4: Training Your PyTorch Model












































