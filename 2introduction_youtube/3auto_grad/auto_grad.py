# https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html

##################################################################################################################################################################
# 1: What Do We Need Autograd For?

##################################################################################################################################################################
# 2: A Simple Example

# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
print(a)

b = torch.sin(a)
# plt.plot(a.detach(), b.detach())

print(b)

c = 2 * b
print(c)

d = c + 1
print(d)

out = d.sum()
print(out)

# Each grad_fn stored with our tensors allows you to walk the computation all the way back to its inputs with its next_functions property. 
print('d:')
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn)

out.backward()
print(a.grad)
# plt.plot(a.detach(), a.grad.detach())

a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
b = torch.sin(a)
c = 2 * b
d = c + 1
out = d.sum()

##################################################################################################################################################################
# 3: Autograd in Training
BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(1000, 100)  # Within a subclass of torch.nn.Module, it’s assumed that we want to track gradients on the layers’ weights for learning. 
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()

print(model.layer2.weight[0][0:10]) # just a small slice
print(model.layer2.weight.grad)  # Within a subclass of torch.nn.Module, it’s assumed that we want to track gradients on the layers’ weights for learning. 

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum()
print(loss)

loss.backward()
print(model.layer2.weight[0][0:10])  # but the weights remain unchanged, because we haven’t run the optimizer yet. The optimizer is responsible for updating model weights based on the computed gradients. 
print(model.layer2.weight.grad[0][0:10])  # We can see that the gradients have been computed for each learning weight; 

optimizer.step()
print(model.layer2.weight[0][0:10])  # You should see that layer2’s weights have changed. 
print(model.layer2.weight.grad[0][0:10])






















