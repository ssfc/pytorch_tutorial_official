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
























































