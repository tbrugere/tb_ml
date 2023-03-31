import functools as ft
import itertools as it # pyright: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combinators import ResidualShortcut
from .basic import MLP

# mostly copied and adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# for some reason, the code in pytorch wasn’t adapted to my use 
# (iirc the way they handle the dimensions is different from the way I do
# or maybe the projections)
# so I had to use this
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / torch.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, input_dim) #get back our original dimension

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        #self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        seq_length, _ = x.size() #unbatched
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(1, 0, 2) # [Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(1, 0, 2) # [SeqLen, Head, Dims]
        values = values.reshape(seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class TransformerBlock(nn.Sequential):
    
    def __init__(self, input_dim, embed_dim=250, activation=nn.LeakyReLU):
        super().__init__()
        
        self.add_module("attention", ResidualShortcut(
            MultiheadAttention(input_dim, embed_dim=embed_dim, num_heads=5)
        ))
        
        self.add_module("norm1", nn.LayerNorm(input_dim))
        
        self.add_module("feed_forward", ResidualShortcut(
            MLP([input_dim, embed_dim, input_dim],norm=None)
        ))
        
        self.add_module("norm2", nn.LayerNorm(input_dim))
