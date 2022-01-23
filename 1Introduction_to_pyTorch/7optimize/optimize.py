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

# Initialize the loss function
# criterion = nn.CrossEntropyLoss()

# 3.2: Optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

##############################################################################################################################################
# 4: Full Implementation
def train_loop(dataloader, model, criterion, optimizer):  # We define train_loop that loops over our optimization code; 
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Forward: compute prediction and loss
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # Call optimizer.zero_grad() to reset the gradients of model parameters. 
        loss.backward()  # Backpropagate the prediction loss with a call to loss.backward(). 
        optimizer.step()  # Update: once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass. 

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, criterion):  # and test_loop that evaluates the model’s performance against our test data; 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter.

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, criterion, optimizer)
    test_loop(test_dataloader, model, criterion)
print("Done!")

















