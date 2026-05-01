from pydantic import BaseModel
import torch

class RWKVConfig(BaseModel):
    vocab_size: int = 50257
    n_layer: int = 6
    n_embed: int = 512
    ctx_len: int = 1024
    dropout: float = 0.0
    bias: bool = False

torch.tensor([1,2,3])
