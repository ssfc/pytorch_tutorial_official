# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html

###############################################################################################################################################################
# 4: Training Your PyTorch Model
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
net.to(device)

# The last ingredients we need are a loss function and an optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # we are doing only 2 training epochs (line 1); 

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):  # Each pass has an inner loop that iterates over the training data (line 4); 
        # (1) prepare data; 
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # (2) forward; 
        outputs = net(inputs)  # we ask the model for its predictions on this batch. 
        loss = criterion(outputs, labels)  # we compute the loss - the difference between outputs (the model prediction) and labels (the correct output).

        # (3) backward; 
        optimizer.zero_grad()  # Call optimizer.zero_grad() to reset the gradients of model parameters. 
        loss.backward()  # we do the backward() pass, and calculate the gradients that will direct the learning. 

        # (4) update; 
        optimizer.step()  # it uses the gradients from the backward() call to nudge the learning weights in the direction it thinks will reduce the loss. 

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():  # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    for data in testloader:
        # (1) prepare data; 
        images, labels = data  
        inputs, labels = inputs.to(device), labels.to(device)

        # (2) forward; 
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))





























