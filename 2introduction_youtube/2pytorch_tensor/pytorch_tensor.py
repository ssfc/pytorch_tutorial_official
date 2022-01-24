# https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

import torch
import math

##################################################################################################################################################
# 1: Creating Tensors
x = torch.empty(3, 4)  # We created a tensor using one of the numerous factory methods attached to the torch module. 
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
























