from pathlib import Path
import torch
from enum import Enum
import rwvk_n_roll

DATAPATH = Path("../../../data/shakes.txt")


class ModelName(Enum):
    BIGRAM = "bigram.pt"
    RNN = "rnn.pt"


MODEL_DIR = Path("output")


def get_model_path(model_name: ModelName):
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / model_name.value


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return device


DEVICE = get_device()
BIGRAM_MODEL_PATH = get_model_path(ModelName.BIGRAM)
RNN_MODEL_PATH = get_model_path(ModelName.RNN)
