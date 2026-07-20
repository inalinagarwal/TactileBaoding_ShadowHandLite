import torch

SRC = "/home/nalin/roto_2/scripts/logs/shadowlite_baoding/rl_only_pt_padtac_bt_ft/2026-07-14_17-12-42/checkpoints/best_agent.pt"
DST = "best_agent_legacy_padtac_bt_ft.pt"
# torch.save(..., _use_new_zipfile_serialization=False)
ckpt = torch.load(SRC, map_location="cpu")
torch.save(ckpt, DST, _use_new_zipfile_serialization=False)
print("wrote", DST)