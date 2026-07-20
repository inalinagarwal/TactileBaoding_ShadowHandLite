import numpy as np, matplotlib.pyplot as plt

JOINTS = ['FFJ4','MFJ4','RFJ4','THJ5','FFJ3','MFJ3','RFJ3','THJ4',
          'FFJ2','MFJ2','RFJ2','FFJ1','MFJ1','RFJ1','THJ2','THJ1']
COUPLED = [(11, 8, 'FF'), (12, 9, 'MF'), (13, 10, 'RF')]   # (J1 idx, J2 idx, name)

sim = np.load("sim_policy_log.npz", allow_pickle=True)
hw  = np.load("hw_replay_log.npz",  allow_pickle=True)
n = min(len(sim["q"]), len(hw["q"]))
scmd, sq = sim["cmd"][:n], sim["q"][:n]
hcmd, hq = hw["cmd"][:n],  hw["q"][:n]

# ---- DIAGNOSTIC: how often did the policy command J1 > J2? ----
print("=== J1 > J2 commanded (what the min() clamp would distort) ===")
for j1, j2, name in COUPLED:
    frac = (scmd[:, j1] > scmd[:, j2]).mean()
    gap  = np.clip(scmd[:, j1] - scmd[:, j2], 0, None)
    print(f"{name}: {100*frac:5.1f}% of steps,  mean overshoot {np.degrees(gap.mean()):.1f} deg")

# ---- per-joint overlay (4x4) ----
fig, ax = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
for j, a in enumerate(ax.flat):
    a.plot(scmd[:, j], 'k--', lw=0.8, label='cmd')
    a.plot(sq[:, j], label='sim')
    a.plot(hq[:, j], label='hw')
    a.set_title(JOINTS[j]); a.grid(True, alpha=0.3)
ax.flat[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("joints_sim_vs_hw.png", dpi=120)

# ---- coupling scatter: J1 vs J2 ----
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
for k, (j1, j2, name) in enumerate(COUPLED):
    ax2[k].scatter(scmd[:, j2], scmd[:, j1], s=5, alpha=0.4, label='sim cmd')
    ax2[k].scatter(hq[:, j2],   hq[:, j1],   s=5, alpha=0.4, label='hw achieved')
    ax2[k].plot([0, 1.57], [0, 1.57], 'k--')          # the J1 = J2 boundary
    ax2[k].set_xlabel(f'{name}J2'); ax2[k].set_ylabel(f'{name}J1'); ax2[k].set_title(name); ax2[k].legend()
fig2.tight_layout(); fig2.savefig("coupling_scatter.png", dpi=120)
print("saved joints_sim_vs_hw.png and coupling_scatter.png")
plt.show()