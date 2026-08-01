import torch

# Resave modern zipfile .pt -> PyTorch 1.4–readable (control laptop).
# Usage: edit SRC/DST, then: python resave_legacy.py

SRC = "/home/nalin/roto_2/best_agent_mass_diff_best.pt"
DST = "best_agent_legacy_mass_diff_best.pt"

ckpt = torch.load(SRC, map_location="cpu")
torch.save(ckpt, DST, _use_new_zipfile_serialization=False)
print("wrote", DST)
