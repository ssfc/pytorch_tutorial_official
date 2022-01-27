####################################################################################################################################################################
# 1: BUILDING MODELS WITH PYTORCH

import torch

class TinyModel(torch.nn.Module):  # This is the PyTorch base class meant to encapsulate behaviors specific to PyTorch Models and their components. 

    def __init__(self):  # This shows the fundamental structure of a PyTorch model: there is an __init__() method that defines the layers and other components of a model; 
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):  # and a forward() method where the computation gets done. 
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')  
print(tinymodel)  # we can print the model, or any of its submodules, to learn about its structure. 

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():  # If a particular Module subclass has learning weights, these weights are expressed as instances of torch.nn.Parameter. 
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():  # The Parameter class is a subclass of torch.Tensor, with the special behavior that when they are assigned as attributes of a Module, they are added to the list of that modules parameters. 
    print(param)

####################################################################################################################################################################
# 2: Common Layer Types
# (1) Linear Layers
lin = torch.nn.Linear(3, 2)  # This is a layer where every input influences every output of the layer to a degree specified by the layer’s weights. 
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)

# (2) Convolutional Layers
import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)  # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # We then pass the output of the convolution through a ReLU activation function (more on activation functions later), then through a max pooling layer. 
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

































