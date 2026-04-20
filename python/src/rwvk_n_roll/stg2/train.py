import torch
from torch import Tensor
from rwvk_n_roll.shared.dataloader import load_data, get_batch
from rwvk_n_roll.stg2.model import RNNModel
from rwvk_n_roll.shared.utils import RNN_MODEL_PATH, DEVICE

batch_size = 32
ctx_len = 8
lr = 1e-3
max_steps = 10000
eval_every = 500

data = load_data()

input_size = 10  # Number of features per time step
hidden_size = 20  # Number of hidden units
output_size = 2  # Number of output classes

model = RNNModel(data.vocab_size).to(DEVICE)

print(f"Model parameters: {model.count_params():,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(max_steps):
    batch = get_batch(data.train_data, batch_size, ctx_len, DEVICE)

    logits: Tensor
    loss: Tensor | None
    logits, loss = model(batch.x, batch.y)
    optimizer.zero_grad(set_to_none=True)

    # none is opt for inference time
    if logits is not None and loss is not None:
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            batch_v = get_batch(data.val_data, batch_size, ctx_len, DEVICE)
            _, val_loss = model(batch_v.x, batch_v.y)
            print(
                f"Step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}"
            )

print("Done!")
torch.save(model.state_dict(), BIGRAM_MODEL_PATH)
