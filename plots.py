#feasibility histogram of J1 - J2 PER FINGER

import glob, numpy as np, matplotlib.pyplot as plt
files = sorted(glob.glob("sim_policy_log_seed*.npz"))
q = np.concatenate([np.load(f, allow_pickle=True)["q"] for f in files])   # (N,16)
COUPLED = [(11, 8, 'FF'), (12, 9, 'MF'), (13, 10, 'RF')]                   # (J1 idx, J2 idx)

fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for k, (j1, j2, name) in enumerate(COUPLED):
    d = np.degrees(q[:, j1] - q[:, j2])
    ax[k].hist(d, bins=60, color="#4c72b0")
    ax[k].axvline(0, color='k', ls='--')
    ax[k].axvspan(0, max(d.max(), 1), alpha=0.15, color='red')            # infeasible
    ax[k].set_title(f"{name}: J1>J2 {100*(d>0).mean():.0f}% infeasible")
    ax[k].set_xlabel("J1 − J2  (deg)")
ax[0].set_ylabel("timesteps")
fig.suptitle("Distal−middle difference (sim achieved): red = hardware-infeasible (J1>J2)")
fig.tight_layout(); fig.savefig("coupling_hist.png", bbox_inches="tight"); plt.show()

#per seed robustness plot

import numpy as np, matplotlib.pyplot as plt
seeds = ['1','2','3']
data = {'FF':[63,81,81], 'MF':[0,1,3], 'RF':[5,23,33]}   # your achieved %s
x = np.arange(3); w = 0.25
fig, ax = plt.subplots(figsize=(7,4))
for i,(f,v) in enumerate(data.items()):
    ax.bar(x+i*w, v, w, label=f)
ax.set_xticks(x+w); ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("J1>J2 achieved (%)"); ax.set_title("Coupling violation by finger across rollouts")
ax.legend(); fig.tight_layout(); fig.savefig("coupling_bars.png", bbox_inches="tight"); plt.show()

#tactile contact raster plot

import numpy as np, matplotlib.pyplot as plt
d = np.load("sim_policy_log_seed1.npz", allow_pickle=True); tac = d["tac"]
names = ("world forearm palm ffknuckle mfknuckle rfknuckle thbase ffprox mfprox rfprox "
         "thprox ffmid mfmid rfmid thhub ffdist mfdist rfdist thmid fftip mftip rftip thdist thtip").split()
active = [i for i in range(24) if tac[:, i].any()]
fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(tac[:, active].T, aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks(range(len(active))); ax.set_yticklabels([names[i] for i in active])
ax.set_xlabel("timestep"); ax.set_title("Tactile contact raster during Baoding (black = contact)")
fig.tight_layout(); fig.savefig("tactile_raster.png", bbox_inches="tight"); plt.show()

