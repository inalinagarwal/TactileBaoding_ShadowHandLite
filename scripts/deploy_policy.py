import torch
import torch.nn as nn


# ---------- MODEL DEFINITIONS ----------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(352, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ELU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),

            nn.Linear(128, 64),
            nn.ELU(),

            nn.Linear(64, 16),
        )

    def forward(self, z):
        return self.policy_net(z)


# ---------- LOAD CHECKPOINT ----------

ckpt = torch.load("/home/nalin/roto/scripts/logs/shadowlite_baoding/rl_only_pt/2026-06-22_16-39-17/checkpoints/best_agent.pt", map_location="cpu")

encoder = Encoder()
policy = Policy()

encoder.load_state_dict(ckpt["encoder"])
policy.load_state_dict(
    {k.replace("policy_net.", ""): v
     for k, v in ckpt["policy"].items()
     if k != "log_std_parameter"},
    strict=False
)

encoder.eval()
policy.eval()


# ---------- TEST WITH DUMMY OBS ----------

dummy_obs = torch.zeros(1, 352)

with torch.no_grad():
    z = encoder(dummy_obs)
    action = policy(z)

print("latent shape:", z.shape)
print("action shape:", action.shape)
print("action:", action)