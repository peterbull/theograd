import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

DATAPATH = Path("../data/shakes.txt")


class Encoder(Protocol):
    def __call__(self, s: str) -> list[int]: ...
class Decoder(Protocol):
    def __call__(self, l: list[int]) -> str: ...


@dataclass(slots=True)
class LoadData:
    train_data: torch.Tensor
    val_data: torch.Tensor
    vocab_size: int
    encode: Encoder
    decode: Decoder


@dataclass(slots=True)
class Batch:
    x: torch.Tensor
    y: torch.Tensor


def load_data(path: Path = DATAPATH, train_spilt=0.9) -> LoadData:
    text: str = ""
    with open(str(path), "r") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(len(data) * train_spilt)
    train_data = data[:n]
    val_data = data[n:]
    return LoadData(train_data, val_data, vocab_size, encode, decode)


def get_batch(data: torch.Tensor, batch_size: int, ctx_len: int, device: str) -> Batch:
    # random starting positions, generate batch size rand idxs
    ix = torch.randint(len(data) - ctx_len, (batch_size,))
    x = torch.stack([data[i : i + ctx_len] for i in ix])
    y = torch.stack([data[i + 1 : i + ctx_len + 1] for i in ix])
    return Batch(x.to(device), y.to(device))


if __name__ == "__main__":
    load_data()
