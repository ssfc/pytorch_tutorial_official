# https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

import torch
import math

##################################################################################################################################################
# 1: Creating Tensors
x = torch.empty(3, 4)  # We created a tensor using one of the numerous factory methods attached to the torch module. 
print(type(x))  # The type of the object returned is torch.Tensor, which is an alias for torch.FloatTensor. 
print(x)  # The torch.empty() call allocates memory for the tensor, but does not initialize it with any values. 




























