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

print(model.layer2.weight.grad[0][0:10])

for i in range(0, 5):
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print(model.layer2.weight.grad[0][0:10])

optimizer.zero_grad()  # After calling optimizer.step(), you need to call optimizer.zero_grad() 

print(model.layer2.weight.grad[0][0:10])

##################################################################################################################################################################
# 4: Turning Autograd Off and On
a = torch.ones(2, 3, requires_grad=True)
print(a)

b1 = 2 * a
print(b1)

a.requires_grad = False
b2 = 2 * a  # When we turn off autograd explicitly with a.requires_grad = False, computation history is no longer tracked, as we see when we compute b2. 
print(b2)

a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = a + b
print(c1)

with torch.no_grad():  # If you only need autograd turned off temporarily, a better way is to use the torch.no_grad(); 
    c2 = a + b

print(c2)

c3 = a * b
print(c3)

def add_tensors1(x, y):
    return x + y

@torch.no_grad()  # torch.no_grad() can also be used as a function or method dectorator; 
def add_tensors2(x, y):
    return x + y


a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = add_tensors1(a, b)
print(c1)

c2 = add_tensors2(a, b)
print(c2)

x = torch.rand(5, requires_grad=True)
y = x.detach()  # detach() method - it creates a copy of the tensor that is detached from the computation history; 

print(x)
print(y)  # matplotlib expects a NumPy array as input, and the implicit conversion from a PyTorch tensor to a NumPy array is not enabled for tensors with requires_grad=True. 

# Autograd and In-place Operations
# a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
# torch.sin_(a)

##################################################################################################################################################################
# 5: Autograd Profiler
device = torch.device('cpu')
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    run_on_gpu = True

x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
    for _ in range(1000):
        z = (z / x) * y

print(prf.key_averages().table(sort_by='self_cpu_time_total'))

##################################################################################################################################################################
# 6: Advanced Topic: More Autograd Detail and the High-Level API















