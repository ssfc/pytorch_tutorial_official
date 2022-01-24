# https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

import torch
import math

##################################################################################################################################################
# 1: Creating Tensors
x = torch.empty(3, 4)  # We created a tensor using one of the numerous factory methods attached to the torch module; all start with torch. 
print(type(x))  # The type of the object returned is torch.Tensor, which is an alias for torch.FloatTensor. 
print(x)  # The torch.empty() call allocates memory for the tensor, but does not initialize it with any values. 

zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)

# (1) Random Tensors and Seeding
torch.manual_seed(1729)  # you’ll want some assurance of the reproducibility of your results; 
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)

# (2) Tensor Shapes
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)  # having the same number of dimensions and the same number of cells in each dimension; 
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)

# The last way to create a tensor that will cover is to specify its data directly from a PyTorch collection: 
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)

# (3) Tensor Data Types
a = torch.ones((2, 3), dtype=torch.int16)  # specifying the tensor’s shape as a series of integer arguments, to grouping those arguments in a tuple; 
print(a)  # printing the tensor also specifies its dtype. 

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)  # converting b to a 32-bit integer with the .to() method. 
print(c)

##################################################################################################################################################
# 2: Math & Logic with PyTorch Tensors
ones = torch.zeros(2, 2) + 1  #  arithmetic operations between tensors and scalars are distributed over every element of the tensor; 
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)

# (1) In Brief: Tensor Broadcasting
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)








