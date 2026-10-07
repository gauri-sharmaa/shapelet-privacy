"""Figures for the paper from results/*.json."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

sw = sorted([json.load(open(f)) for f in glob.glob("results/sweep_k*.json")], key=lambda r: r["k"])
fig, ax = plt.subplots(figsize=(3.3, 1.9))
ks = [r["k"] for r in sw]
ax.plot(ks, [r["n_unique_customers"] for r in sw], "o-", color="#1f5fa8", ms=3, label="customers singled out")
ax.plot(ks, [r["n_unique_suspicious"] for r in sw], "s--", color="#c0392b", ms=3, label="…of which suspicious")
ax.set_xscale("log"); ax.set_xlabel("released shapelets $k$"); ax.set_ylabel("customers")
ax2 = ax.twinx(); ax2.plot(ks, [r["detection"]["pr_auc"] for r in sw], "^:", color="gray", ms=3)
ax2.set_ylabel("PR-AUC", color="gray"); ax2.spines["top"].set_visible(False)
ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(); fig.savefig("paper/fig_sweep.pdf")

dfn = [json.load(open(f)) for f in sorted(glob.glob("results/defenses_seed*.json"))]
base = [json.load(open(f)) for f in sorted(glob.glob("results/baselines_seed*.json"))]
fig, ax = plt.subplots(figsize=(3.3, 1.9))
lab = {"none": "as trained", "noise-0.1": "σ=.1", "noise-0.25": "σ=.25", "noise-0.5": "σ=.5", "noise-1.0": "σ=1",
       "kanon-5": "k=5", "kanon-20": "k=20", "kanon-50": "k=50"}
for k, l in lab.items():
    x = np.mean([r[k]["reid"]["top1"] for r in dfn]) * 100; y = np.mean([r[k]["detection"]["pr_auc"] for r in dfn])
    c = "#c0392b" if k == "none" else ("#1f5fa8" if k.startswith("noise") else "#7d3c98")
    ax.scatter(x, y, s=12, color=c)
    if k.startswith("kanon"):
        if k == "kanon-20":
            ax.annotate("k-anon. (k=5,20,50)", (x, y), fontsize=6.5, xytext=(-30, 8), textcoords="offset points")
    elif k in ("noise-0.5", "noise-1.0"):
        if k == "noise-0.5":
            ax.annotate("noise σ≥.5", (x, y), fontsize=6.5, xytext=(2, 8), textcoords="offset points")
    else:
        off = (2, -10) if k == "noise-0.1" else (2, 2)
        ax.annotate(l, (x, y), fontsize=6.5, xytext=off, textcoords="offset points")
ly = np.mean([r["learned"]["detection"]["pr_auc"] for r in dfn])
ax.scatter(0, ly, marker="*", s=40, color="#27ae60"); ax.annotate("learned", (0, ly), fontsize=6.5, xytext=(3, -6), textcoords="offset points")
cb = np.mean([b["catboost"]["pr_auc"] for b in base]); ax.axhline(cb, ls="--", lw=0.7, color="gray")
ax.text(10, cb + 0.002, "CatBoost (no shapelets)", fontsize=6, color="gray")
ax.set_ylim(0.86, 0.995)
ax.set_xlabel("top-1 re-identification (%)"); ax.set_ylabel("PR-AUC")
fig.tight_layout(); fig.savefig("paper/fig_tradeoff.pdf")
print("figs ok")

# Realistic adversary: correct re-identifications vs adversary's view (all accounts, ratio 2)
R = [json.load(open(f)) for f in sorted(glob.glob("results/realistic_seed*.json"))]
views = ["exact", "amounts ±5%", "80% of txns, ±5%", "50% of txns, ±5%"]
labels = ["exact", "±5%", "80%\n+±5%", "50%\n+±5%"]
fig, axs = plt.subplots(1, 2, figsize=(3.3, 2.0))
for m, lab, col, mk in [("rdst", "RDST", "#c0392b", "o"), ("rdst_normonly", "RDST, no stats", "#e67e22", "s"), ("stc", "shapelet transform", "#1f5fa8", "^")]:
    c = [np.mean([r[m][f"{v}|pool=1.0|ratio=2.0"]["customers_exposed"] for r in R if r[m]]) for v in views]
    p = [100 * np.mean([r[m][f"{v}|pool=1.0|ratio=2.0"]["precision"] for r in R if r[m]]) for v in views]
    axs[0].plot(range(4), c, marker=mk, ms=3, color=col, label=lab)
    axs[1].plot(range(4), p, marker=mk, ms=3, color=col)
for a, yl in zip(axs, ["customers re-identified", "precision (%)"]):
    a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=6); a.set_ylabel(yl, fontsize=7)
axs[1].set_ylim(0, 105)
axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=5.5, loc="lower left")
for a in axs: a.set_xlabel("adversary's view (txns seen)", fontsize=6)
fig.tight_layout(); fig.savefig("paper/fig_real.pdf")
print("fig_real ok")
