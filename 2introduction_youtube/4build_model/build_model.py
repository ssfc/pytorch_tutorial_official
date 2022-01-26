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




































