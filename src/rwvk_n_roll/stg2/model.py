import torch
import torch.nn as nn
from torch.nn import functional as F


class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, hidden_size: int):
        super().__init__()

        self.table = nn.Embedding(vocab_size, n_embed)
        self.linear1 = nn.Linear(n_embed + hidden_size, hidden_size)
        self.tanh = nn.Tanh
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, targets=None):
        (
            B,
            T,
        ) = idx.shape

        # TODO: initialize hidden state to zeros
        # shape: (B, hidden_size)

        # TODO: embed all tokens at once
        # shape after: (B, T, n_embd)

        logits = []

        for t in range(T):
            continue
            # TODO: get the embedding for the current timestep t
            # shape: (B, n_embd)

            # TODO: concatenate current input with previous hidden state
            # shape: (B, n_embd + hidden_size)

            # TODO: compute new hidden state using the linear layer + activation

            # TODO: compute logit for this timestep from the hidden state
            # shape: (B, vocab_size)

            # TODO: append logit to logits list

        # TODO: stack logits list into a tensor
        # shape: (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # TODO: compute cross entropy loss (same as bigram stage)

        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
