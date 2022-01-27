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

# What is a 3D tensor anyway? Think about it like this. If you have a vector, indexing into the vector gives you a scalar. If you have a matrix, indexing into the matrix gives you a vector. If you have a 3D tensor, then indexing into the tensor gives you a matrix! 

# Index into V and get a scalar (0 dimensional tensor)
print(V[0])
# Get a Python number from it
print(V[0].item())  # Returns the value of this tensor as a standard Python number. This only works for tensors with one element. 

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])

x = torch.randn((3, 4, 5))
print(x)

# (2) Operations with Tensors
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(z)

# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# second arg specifies which axis to concat along
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# If your tensors are not compatible, torch will complain.  Uncomment to see the error
# torch.cat([x_1, x_2])

# (3) Reshaping Tensors: Use the .view() method to reshape a tensor. 
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
# Same as above.  If one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))

#################################################################################################################################################################
# 2: Computation Graphs and Automatic Differentiation

# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previously could; 
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)

# If requires_grad=True, the Tensor object keeps track of how it was created. 
print(z.grad_fn)

# Let's sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)  # compute gradient for x; 



































