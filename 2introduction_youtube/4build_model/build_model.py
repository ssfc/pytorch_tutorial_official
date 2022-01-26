####################################################################################################################################################################
# 1: BUILDING MODELS WITH PYTORCH

import torch

class TinyModel(torch.nn.Module):  # This is the PyTorch base class meant to encapsulate behaviors specific to PyTorch Models and their components. 

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():  # If a particular Module subclass has learning weights, these weights are expressed as instances of torch.nn.Parameter. 
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)





































