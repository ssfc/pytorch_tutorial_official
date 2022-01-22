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
loss.backward()
print(w.grad)
print(b.grad)

###################################################################################################
# 3: Disabling Gradient Tracking
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

###################################################################################################
# 4: More on Computational Graphs

###################################################################################################
# 5: Optional Reading: Tensor Gradients and Jacobian Products
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
























