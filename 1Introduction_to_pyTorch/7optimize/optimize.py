# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# (1) makes a guess about the output, 
# (2) calculates the error in its guess (loss), 
# (3) collects the derivatives of the error with respect to its parameters (as we saw in the previous section), 
# (4) and optimizes these parameters using gradient descent; 

##############################################################################################################################################
# 1: Prerequisite Code
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

##############################################################################################################################################
# 2: Hyperparameters
learning_rate = 1e-3  # Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
batch_size = 64  # Batch Size - the number of data samples propagated through the network before the parameters are updated; 
epochs = 5  # Each iteration of the optimization loop is called an epoch; 

##############################################################################################################################################
# 3: Optimization Loop
# 3.1: Loss Function


























