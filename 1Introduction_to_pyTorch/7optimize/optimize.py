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
# loss_fn = nn.CrossEntropyLoss()

# 3.2: Optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

##############################################################################################################################################
# 4: Full Implementation
def train_loop(dataloader, model, loss_fn, optimizer):  # We define train_loop that loops over our optimization code; 
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):  # and test_loop that evaluates the model’s performance against our test data; 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# We initialize the loss function and optimizer, and pass it to train_loop and test_loop. 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

















