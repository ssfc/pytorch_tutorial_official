# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# (1) makes a guess about the output, 
# (2) calculates the error in its guess (loss), 
# (3) collects the derivatives of the error with respect to its parameters (as we saw in the previous section), 
# (4) and optimizes these parameters using gradient descent; 

##############################################################################################################################################
# 1: Prerequisite Code
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)


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
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 10)
    

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

model = NeuralNetwork()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
model.to(device)

##############################################################################################################################################
# 2: Hyperparameters
learning_rate = 1e-3  # Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
batch_size = 64  # Batch Size - the number of data samples propagated through the network before the parameters are updated; 
EPOCHS = 5  # Each iteration of the optimization loop is called an epoch; 

##############################################################################################################################################
# 3: Optimization Loop
# 3.1: Loss Function

# Initialize the loss function
# criterion = nn.CrossEntropyLoss()

# 3.2: Optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

##############################################################################################################################################
# 4: Full Implementation
def train_loop(dataloader, model, criterion, optimizer):  # We define train_loop that loops over our optimization code; 
    size = len(dataloader.dataset)
    for i, data in enumerate(dataloader):
        # (1) prepare data; 
        input, target = data
        input, target = input.to(device), target.to(device)
        # (2) Forward
        pred = model(input)  # we ask the model for its predictions on this batch. 
        loss = criterion(pred, target)  # we compute the loss - the difference between outputs (the model prediction) and labels (the correct output).

        # (3) Backpropagation
        optimizer.zero_grad()  # Call optimizer.zero_grad() to reset the gradients of model parameters. 
        loss.backward()  # we do the backward() pass, and calculate the gradients that will direct the learning. 

        # (4) update
        optimizer.step()  # it uses the gradients from the backward() call to nudge the learning weights in the direction it thinks will reduce the loss. 

        if i % 100 == 0:
            loss, current = loss.item(), i * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, criterion):  # and test_loop that evaluates the model’s performance against our test data; 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)
            pred = model(input)
            test_loss += criterion(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter.

EPOCHS = 10

'''
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, model, criterion, optimizer)
    test_loop(test_dataloader, model, criterion)
print("Done!")
'''


for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for i, data in enumerate(train_dataloader):
        # (1) prepare data; 
        input, target = data
        input, target = input.to(device), target.to(device)
        # (2) Forward
        pred = model(input)  # we ask the model for its predictions on this batch. 
        loss = criterion(pred, target)  # we compute the loss - the difference between outputs (the model prediction) and labels (the correct output).

        # (3) Backpropagation
        optimizer.zero_grad()  # Call optimizer.zero_grad() to reset the gradients of model parameters. 
        loss.backward()  # we do the backward() pass, and calculate the gradients that will direct the learning. 

        # (4) update
        optimizer.step()  # it uses the gradients from the backward() call to nudge the learning weights in the direction it thinks will reduce the loss. 

        if i % 100 == 0:
            loss, current = loss.item(), i * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")














