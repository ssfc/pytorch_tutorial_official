# https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html#introduction-to-torch-s-tensor-library

#################################################################################################################################################################
# 1: Introduction to Torch’s tensor library



# Author: Robert Guthrie

import torch

torch.manual_seed(1)

# (1) Creating Tensors

# torch.tensor(data) creates a torch.Tensor object with the given data.
V_data = [1., 2., 3.]
V = torch.tensor(V_data)  # Tensors can be created from Python lists with the torch.tensor() function. 
print(V)

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2.
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.tensor(T_data)
print(T)





















































