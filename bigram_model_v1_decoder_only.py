import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
head_size = n_embd // n_head
# ------------

batch_size = 32 # B
block_size = 32 # T
n_embd = 64 # C
#head_size = 4
dropout = 0.5
intermediate = 4 * n_embd

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, _, C = x.shape
        key = self.key(x) 
        query = self.query(x)
        #print(key.shape, query.shape)
        wei = query @ key.transpose(-2, -1) * C**-0.5 # B, T, H * B, H ,T = B, T, T
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1) # B, T, T 
        v = self.value(x) # B, T, H
        out = wei @ v # B, T, T X B, T, H = B, T, H
        return out

class MultiHeadAttention(nn.Module):

  def __init__(self):
    super().__init__()
    self.num_heads = n_head
    self.embed_dim = n_embd
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      # x: B, T, C
      y = torch.cat([head(x) for head in self.heads], dim=-1)
      print(y.shape)
      y = self.dropout(self.proj(y))
      return y


class FeedForward(nn.Module):

  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(n_embd, intermediate)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(intermediate, n_embd)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x): 
    # X: B, T, C
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    return self.dropout(x)

class Block(nn.Module):

  def __init__(self):
    super().__init__()
    self.self_attention = MultiHeadAttention()
    self.feed_forward = FeedForward()
  

  def forward(self, x):
    # estabilish residual connections
    x = x + self.self_attention(x)
    x = x + self.feed_forward(x)
    return x

class LanguageModelBigram(nn.Module):

  # bring all models together
  def __init__(self, vocab_size, n_embd, n_head):
    super().__init__()
    self.vocab_size = vocab_size
    self.num_heads = nheads
    self.embed_dim = embed_dim
    self.embedding_layer = nn.Embedding(self.vocab_size, n_embd)
    self.pos_encoding = nn.Embedding(block_size, n_embd)

  def forward(self):
    