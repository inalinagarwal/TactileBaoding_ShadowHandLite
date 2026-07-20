import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

names = ("world forearm palm ffknuckle mfknuckle rfknuckle thbase "
         "ffprox mfprox rfprox thprox ffmid mfmid rfmid thhub "
         "ffdist mfdist rfdist thmid fftip mftip rftip thdist thtip").split()

# ---- load + pool rollouts ----
files = sorted(glob.glob("sim_policy_log_seed*.npz"))
tac = np.concatenate([np.load(f, allow_pickle=True)["tac"] for f in files], axis=0)  # (steps, 24)
freq = 100.0 * tac.mean(0)
n_steps = tac.shape[0]

# ---- console table ----
print(f"{n_steps} steps pooled across {len(files)} rollouts\n")
for i in np.argsort(-freq):
    print(f"{i:2d} {names[i]:10s} {freq[i]:5.1f}%  {'#' * int(freq[i] / 2)}")

# ---- colour by finger group ----
def group(name):
    if name.startswith("ff"): return ("First finger", "#1f77b4")
    if name.startswith("mf"): return ("Middle finger", "#ff7f0e")
    if name.startswith("rf"): return ("Ring finger", "#2ca02c")
    if name.startswith("th"): return ("Thumb", "#d62728")
    return ("Palm / other", "#7f7f7f")

# ---- sorted horizontal bar plot ----
order  = np.argsort(freq)                     # ascending -> largest on top in barh
labels = [names[i] for i in order]
vals   = freq[order]
colors = [group(names[i])[1] for i in order]

fig, ax = plt.subplots(figsize=(8, 9))
bars = ax.barh(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Contact frequency (% of timesteps in contact)", fontsize=12)
ax.set_title("Per-link tactile activation during Baoding\n"
             f"Shadow Hand Lite, blind (prop+tactile) policy — {n_steps} steps", fontsize=12)
ax.set_xlim(0, max(vals) * 1.15 + 1)
ax.margins(y=0.01)

for b, v in zip(bars, vals):
    if v > 0:
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8)

seen = {}
for i in order:
    g, c = group(names[i]); seen[g] = c
handles = [Patch(facecolor=c, edgecolor="black", label=g) for g, c in seen.items()]
ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)

ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("tactile_activation.png", dpi=300, bbox_inches="tight")
fig.savefig("tactile_activation.pdf", bbox_inches="tight")     # vector, for LaTeX
print("\nsaved tactile_activation.png / .pdf")
plt.show()