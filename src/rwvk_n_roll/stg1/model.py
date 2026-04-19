import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None) -> tuple[Tensor, Tensor | None]:
        logits: Tensor = self.table(idx)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
