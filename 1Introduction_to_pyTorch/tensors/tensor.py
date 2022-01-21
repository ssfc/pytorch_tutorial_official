# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

import torch
import numpy as np

#######################################################################################################
# 1: Initializing a Tensor
# Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


















