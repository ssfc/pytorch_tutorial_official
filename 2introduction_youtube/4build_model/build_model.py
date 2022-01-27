####################################################################################################################################################################
# 1: BUILDING MODELS WITH PYTORCH

import torch

class TinyModel(torch.nn.Module):  # This is the PyTorch base class meant to encapsulate behaviors specific to PyTorch Models and their components. 

    def __init__(self):  # This shows the fundamental structure of a PyTorch model: there is an __init__() method that defines the layers and other components of a model; 
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):  # and a forward() method where the computation gets done. 
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')  
print(tinymodel)  # we can print the model, or any of its submodules, to learn about its structure. 

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():  # If a particular Module subclass has learning weights, these weights are expressed as instances of torch.nn.Parameter. 
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():  # The Parameter class is a subclass of torch.Tensor, with the special behavior that when they are assigned as attributes of a Module, they are added to the list of that modules parameters. 
    print(param)

####################################################################################################################################################################
# 2: Common Layer Types
# (1) Linear Layers
lin = torch.nn.Linear(3, 2)  # This is a layer where every input influences every output of the layer to a degree specified by the layer’s weights. 
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)

# (2) Convolutional Layers
import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)  # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # We then pass the output of the convolution through a ReLU activation function (more on activation functions later), then through a max pooling layer. 
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# (3) Recurrent Layers
class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)  # The input will be a sentence with the words represented as indices of one-hot vectors. The embedding layer will then map these down to an embedding_dim-dimensional space. 
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))  # The LSTM takes this sequence of embeddings and iterates over it, fielding an output vector of length hidden_dim. 
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))  # The final linear layer acts as a classifier; 
        tag_scores = F.log_softmax(tag_space, dim=1)  # applying log_softmax() to the output of the final layer converts the output into a normalized set of estimated probabilities that a given word maps to a given tag. 
        return tag_scores

# SEQUENCE MODELS AND LONG SHORT-TERM MEMORY NETWORKS
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html





























