"""Use a trained model in a notebook or a script, run from the project folder."""

import torch

import lighter

model = lighter.load_model("checkpoints/last.ckpt")  # the checkpoint of recipes/resume.yaml
model.eval()
with torch.no_grad():
    print(model(torch.tensor([[0.5, -1.0]])))  # about 2 * 0.5 - 3 * -1.0 + 0.5 = 4.5
