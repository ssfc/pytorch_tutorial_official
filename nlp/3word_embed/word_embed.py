# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

##################################################################################################################################################################
# 1: Getting Dense Word Embeddings

##################################################################################################################################################################
# 2: Word Embeddings in Pytorch

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print("hello embedding: ", hello_embed)

##################################################################################################################################################################
# 3: An Example: N-Gram Language Modeling

CONTEXT_SIZE = 2  # window size; 
EMBEDDING_DIM = 10

# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

# Print the first 3, just so you can see what they look like.
print("ngrams[:3]: ", ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
criterion = nn.NLLLoss()

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
print("device: ", device)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:
        # (1) Prepare data
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)  # model.zero_grad() and optimizer.zero_grad() are the same IF all your model parameters are in that optimizer. I found it is safer to call model.zero_grad() to make sure all grads are zero, e.g. if you have two or more optimizers for one model.
        target_idxs = torch.tensor([word_to_ix[target]], dtype=torch.long)
        context_idxs, target_idxs = context_idxs.to(device), target_idxs.to(device)

        # (2) Forward 
        log_probs = model(context_idxs)  # getting log probabilities over next words
        loss = criterion(log_probs, target_idxs)  # Compute your loss function. (Again, Torch wants the target word wrapped in a tensor)

        # (3) Backward
        optimizer.zero_grad()
        loss.backward()

        # (4) Update the gradient
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print("losses:", losses)  # The loss decreased every iteration over the training data!

# To get the embedding of a particular word, e.g. "beauty"
print("Embedding beauty: ", model.embeddings.weight[word_to_ix["beauty"]])

##################################################################################################################################################################
# 3: Exercise: Computing Word Embeddings: Continuous Bag-of-Words




















































