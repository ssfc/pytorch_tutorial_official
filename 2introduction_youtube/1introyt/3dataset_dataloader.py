# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html

###############################################################################################################################################################
# 3: Datasets and Dataloaders

#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),  # transforms.ToTensor() converts images loaded by Pillow into PyTorch tensors. 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transforms.Normalize() adjusts the values of the tensor so that their average is zero and their standard deviation is 0.5. 


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)  # organizes the input tensors served by the Dataset into batches with the parameters you specify.










