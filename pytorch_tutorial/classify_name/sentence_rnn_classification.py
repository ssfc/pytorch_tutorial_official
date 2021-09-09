# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#exercises
# we print our type after original codes;

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --------------------------- get all files -------------------------
def find_files(path):
    return glob.glob(path)


print("files found: ", find_files('data/names/*.txt'))
print("files found: ", find_files('data/sentences/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)  # equals to len(vocabulary)
print("all letters: ", all_letters)

# ------------------------ get all words (vocabulary) ---------------------
with open("data/sentences/priority_corpus.txt", "r", encoding='UTF-8') as f:
    data = f.readlines()

sentences = []
for line in data:
    sentences.append(line[1: len(line) - 2])

print("Sentences: ", sentences)
print("Size of sentences: ", len(sentences))


def tokenize_sentence():  # split each sentence into list, made up with words;
    tokens = [x.split("*") for x in sentences]

    return tokens


tokenized_sentence = tokenize_sentence()  # split each sentence into list, made up with words;
print("Tokenized sentence: ", tokenized_sentence)

vocabulary = []
for sentence in tokenized_sentence:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)  # add word not in vocabulary into vocabulary;

word_size = len(vocabulary)
print("Vocabulary: ", vocabulary)
print("Size of vocabulary: ", word_size)


# ----------------------------------------------------------------------------------------------------
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print("unicode to ascii: ", unicode_to_ascii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []
type_lines = {}
all_types = []


# Read a file and split into lines
def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def read_type_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]


for file_name in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(file_name))[0]
    all_categories.append(category)
    lines = read_lines(file_name)
    category_lines[category] = lines

for file_name in find_files('data/sentences/*.txt'):
    sentence_type = os.path.splitext(os.path.basename(file_name))[0]
    all_types.append(sentence_type)
    lines = read_type_lines(file_name)
    type_lines[sentence_type] = lines

all_types.pop(0)  # remove the first element;

n_categories = len(all_categories)
print("all categories: ", all_categories)  # method, relation, definition;
# print("category lines: ", category_lines)
print("all types: ", all_types)  # method, relation, definition;

######################################################################
# Now we have ``category_lines``, a dictionary mapping each category
# (language) to a list of lines (names). We also kept track of
# ``all_categories`` (just a list of languages) and ``n_categories`` for
# later reference.
#

print("First 5 words of Italian: ", category_lines['Italian'][:5])
print("First 5 sentences of corpus: ", type_lines['priority_corpus'][:3])


######################################################################
# Turning Names into Tensors
# --------------------------
#
# Now that we have all the names organized, we need to turn them into
# Tensors to make any use of them.
#
# To represent a single letter, we use a "one-hot vector" of size
# ``<1 x n_letters>``. A one-hot vector is filled with 0s except for a 1
# at index of the current letter, e.g. ``"b" = <0 1 0 0 0 ...>``.
#
# To make a word we join a bunch of those into a 2D matrix
# ``<line_length x 1 x n_letters>``.
#
# That extra 1 dimension is because PyTorch assumes everything is in
# batches - we're just using a batch size of 1 here.
#


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


print("letter to index: ")
print("J: ", letter_to_index('J'))
print("o: ", letter_to_index('o'))

# same as function letter_to_index;
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}  # create dictionary;
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}  # create dictionary;

print("word to index: ", word2idx)
print("word 23 to index: ", word2idx['23'])


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


print("letter to tensor: ")
print(letter_to_tensor('J'))  # 1 appear at position 35;


def word_to_tensor(word):
    tensor = torch.zeros(1, len(vocabulary))
    tensor[0][word2idx[word]] = 1
    return tensor


print("word to tensor: ")
print(word_to_tensor('23'))  # 1 appear at position 35;


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


print("line to tensor: ")
print(line_to_tensor('Jones').size())


def sentence_to_tensor(this_sentence):
    tokens = this_sentence.split("*")[1:-1]
#    tokens = this_sentence.split("*")
    tensor = torch.zeros(len(tokens), 1, len(vocabulary))
    for count, word in enumerate(tokens):
        tensor[count][0][word2idx[word]] = 1

    return tensor


print("sentence to tensor: ", sentence_to_tensor("*5*23*17*72*72*72*72*5*38*38*38*23*23*1*").size())
print("sentence to tensor: ", sentence_to_tensor("*5*23*17*72*72*72*72*5*38*38*38*23*23*1*"))

######################################################################
# Creating the Network
# ====================
#
# Before autograd, creating a recurrent neural network in Torch involved
# cloning the parameters of a layer over several timesteps. The layers
# held hidden state and gradients which are now entirely handled by the
# graph itself. This means you can implement a RNN in a very "pure" way,
# as regular feed-forward layers.
#
# This RNN module (mostly copied from `the PyTorch for Torch users
# tutorial <https://pytorch.org/tutorials/beginner/former_torchies/
# nn_tutorial.html#example-2-recurrent-net>`__)
# is just 2 linear layers which operate on an input and hidden state, with
# a LogSoftmax layer after the output.
#
# .. figure:: https://i.imgur.com/Z2xbySO.png
#    :alt:
#
#


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
rnn_ssfc = RNN(len(vocabulary), n_hidden, len(all_types))

######################################################################
# To run a step of this network we need to pass an input (in our case, the
# Tensor for the current letter) and a previous hidden state (which we
# initialize as zeros at first). We'll get back the output (probability of
# each language) and a next hidden state (which we keep for the next
# step).
#

input = letter_to_tensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)
print("single letter output: ", output.size(), output)

input = word_to_tensor('23')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn_ssfc(input, hidden)
print("single word output: ", output.size(), output)

######################################################################
# For the sake of efficiency we don't want to be creating a new Tensor for
# every step, so we will use ``line_to_tensor`` instead of
# ``letter_to_tensor`` and use slices. This could be further optimized by
# pre-computing batches of Tensors.
#

input_line = line_to_tensor('Albert')
hidden_line = torch.zeros(1, n_hidden)

output_line, next_hidden = rnn(input_line[0], hidden_line)
print("line output: ", output_line.size(), output_line)


input_sentence = sentence_to_tensor('*5*23*17*72*72*72*72*5*38*38*38*23*23*1*')
hidden = torch.zeros(1, n_hidden)
#print("input_sentence[0]: ", input_sentence[0].size())

#output, next_hidden = rnn(input_sentence[0], hidden)
#print("sentence output: ", output)

######################################################################
# As you can see the output is a ``<1 x n_categories>`` Tensor, where
# every item is the likelihood of that category (higher is more likely).
#


######################################################################
#
# Training
# ========
# Preparing for Training
# ----------------------
#
# Before going into training we should make a few helper functions. The
# first is to interpret the output of the network, which we know to be a
# likelihood of each category. We can use ``Tensor.topk`` to get the index
# of the greatest value:
#

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


print(categoryFromOutput(output))


######################################################################
# We will also want a quick way to get a training example (a name and its
# language):
#


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

######################################################################
# Training the Network
# --------------------
#
# Now all it takes to train this network is show it a bunch of examples,
# have it make guesses, and tell it if it's wrong.
#
# For the loss function ``nn.NLLLoss`` is appropriate, since the last
# layer of the RNN is ``nn.LogSoftmax``.
#

criterion = nn.NLLLoss()

######################################################################
# Each loop of training will:
#
# -  Create input and target tensors
# -  Create a zeroed initial hidden state
# -  Read each letter in and
#
#    -  Keep hidden state for next letter
#
# -  Compare final output to target
# -  Back-propagate
# -  Return the output and loss
#

learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


######################################################################
# Now we just have to run that with a bunch of examples. Since the
# ``train`` function returns both the output and loss we can print its
# guesses and also keep track of loss for plotting. Since there are 1000s
# of examples we print only every ``print_every`` examples, and take an
# average of the loss.
#


n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

######################################################################
# Plotting the Results
# --------------------
#
# Plotting the historical loss from ``all_losses`` shows the network
# learning:
#


plt.figure()
plt.plot(all_losses)

######################################################################
# Evaluating the Results
# ======================
#
# To see how well the network performs on different categories, we will
# create a confusion matrix, indicating for every actual language (rows)
# which language the network guesses (columns). To calculate the confusion
# matrix a bunch of samples are run through the network with
# ``evaluate()``, which is the same as ``train()`` minus the backprop.
#

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


######################################################################
# You can pick out bright spots off the main axis that show which
# languages it guesses incorrectly, e.g. Chinese for Korean, and Spanish
# for Italian. It seems to do very well with Greek, and very poorly with
# English (perhaps because of overlap with other languages).
#


######################################################################
# Running on User Input
# ---------------------
#

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(line_to_tensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Dovesky')
predict('Jackson')
predict('Satoshi')
