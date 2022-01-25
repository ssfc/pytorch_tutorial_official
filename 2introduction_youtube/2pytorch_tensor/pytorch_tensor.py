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

# (2) More Math with Tensors
# common functions
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# bitwise operations
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# comparisons:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

# reductions:
print('\nReduction ops:')
print(torch.max(d))        # returns a single-element tensor
print(torch.max(d).item()) # extracts the value from the returned tensor
print(torch.mean(d))       # average
print(torch.std(d))        # standard deviation
print(torch.prod(d))       # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # filter unique elements

# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.])         # x unit vector
v2 = torch.tensor([0., 1., 0.])         # y unit vector
m1 = torch.rand(2, 2)                   # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # 3 times m1
print(torch.svd(m3))       # singular value decomposition

# (3) Altering Tensors in Place: most of the math functions have a version with an appended underscore (_) that will alter a tensor in place. 
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # this operation creates a new tensor in memory
print(a)              # a has not changed

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # note the underscore
print(b)              # b has changed

# Note that these in-place arithmetic functions are methods on the torch.Tensor object, not attached to the torch module like many other functions (e.g., torch.sin()). 
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)


a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print("c: ", c)
d = torch.matmul(a, b, out=c)  # have an out argument that lets you specify a tensor to receive the output. 
print("c: ", c)                # contents of c have changed

assert c is d           # test c & d are same object, not just containing equal values
assert id(c), old_id    # make sure that our new c is the same object as the old one

torch.rand(2, 2, out=c) # works for creation too!
print("c: ", c)                # c has changed again
assert id(c), old_id    # still the same object!

##################################################################################################################################################
# 3: Copying Tensors
a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)       # ...and b is also altered

# But what if you want a separate copy of the data to work on? The clone() method is there for you:
a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561          # a changes...
print(b)               # ...but b is still all ones

a = torch.rand(2, 2, requires_grad=True) # turn on autograd: requires_grad=True - this means that autograd and computation history tracking are turned on. 
print(a)

b = a.clone()
print(b)  # we can see that it’s tracking its computation history - it has inherited a’s autograd settings, and added to the computation history. 

c = a.detach().clone()  # The detach() method detaches the tensor from its computation history. It says, “do whatever comes next as if autograd was off.”
print(c)  # we see no computation history, and no requires_grad=True. 

print(a)  # It does this without changing a - you can see that when we print a again at the end, it retains its requires_grad=True property. 

##################################################################################################################################################
# 4: moving to GPU 
if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')

if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')  # There are multiple ways to get your data onto your target device. You may do it at creation time; 
    print(gpu_rand)
else:
    print('Sorry, CPU only.')


if torch.cuda.is_available():
    my_device = torch.device('cuda')  # By default, new tensors are created on the CPU, so we have to specify when we want to create our tensor on the GPU with the optional device argument. 
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))  # creating a device handle that can be passed to your tensors instead of a string; 

x = torch.rand(2, 2, device=my_device)
print(x)

y = torch.rand(2, 2)  #  creates a tensor on CPU, and moves it to whichever device handle you acquired in the previous cell.
y = y.to(my_device)

##################################################################################################################################################
# 5: Manipulating Tensor Shapes
 













