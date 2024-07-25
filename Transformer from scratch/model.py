import torch 
import torch.nn as nn 
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int): # dimention of the model - 512, vocabulary size
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEnconding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: 
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) 
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # add a dimention for batch (1, seq_len, d_model)

        self.register_buffer('pe', pe) # save the pe

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad(False) # do not recalculate this tensors that is why we use the requires_grad
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(0)) # Added 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # -1 for the last dimension(everything after the batch)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std - self.eps) + self.bias


class FeedForward(nn.Module): 

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: 
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
    
    def forward(self, x):
        # (batch, seq_len, d_model) ---> (batch, seq_len, d_ff) ---> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()