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



















