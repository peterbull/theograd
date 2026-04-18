import torch
from dataloader import load_data
from model import BigramModel
from utils import MODEL_PATH, DEVICE

data = load_data()
encode = data.encode
decode = data.decode

model = BigramModel(data.vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def generate(prompt="", max_new_tokens=200, temperature=1.0):
    if prompt:
        idx = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())


print(generate("ROMEO: "))
