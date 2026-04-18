from pathlib import Path
import torch

MODEL_DIR = Path("output")
MODEL_NAME = "bigram.pt"


def get_model_path():
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / MODEL_NAME


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return device


DEVICE = get_device()
MODEL_PATH = get_model_path()
