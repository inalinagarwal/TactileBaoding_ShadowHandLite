import glob, numpy as np

COUPLED = [(11, 8, 'FF'), (12, 9, 'MF'), (13, 10, 'RF')]   # (J1 idx, J2 idx, name)
files = sorted(glob.glob("sim_policy_log_seed*.npz"))
print("rollouts found:", files, "\n")

agg = {name: [] for _, _, name in COUPLED}
for f in files:
    sq = np.load(f, allow_pickle=True)["q"]          # achieved joint pos (steps, 16)
    print(f"--- {f}  ({len(sq)} steps) ---")
    for j1, j2, name in COUPLED:
        frac = (sq[:, j1] > sq[:, j2]).mean()
        agg[name].append(frac)
        print(f"   {name}: J1>J2 achieved {100*frac:5.1f}%")
    print()

print("=== aggregate across rollouts (achieved J1 > J2) ===")
for _, _, name in COUPLED:
    a = np.array(agg[name]) * 100
    print(f"{name}: mean {a.mean():5.1f}%   per-run {[f'{x:.0f}%' for x in a]}")