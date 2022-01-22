# https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output, true label; 
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b  # predicted label; 
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

###################################################################################################
# 1: Tensors, Functions and Computational graph
print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

###################################################################################################
# 2: Computing Gradients

































