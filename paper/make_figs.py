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
    ax.scatter(x, y, s=12, color=c); ax.annotate(l, (x, y), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
ly = np.mean([r["learned"]["detection"]["pr_auc"] for r in dfn])
ax.scatter(0, ly, marker="*", s=40, color="#27ae60"); ax.annotate("learned", (0, ly), fontsize=6.5, xytext=(3, -6), textcoords="offset points")
cb = np.mean([b["catboost"]["pr_auc"] for b in base]); ax.axhline(cb, ls="--", lw=0.7, color="gray")
ax.text(10, cb + 0.002, "CatBoost (no shapelets)", fontsize=6, color="gray")
ax.set_ylim(0.86, 0.995)
ax.set_xlabel("top-1 re-identification (%)"); ax.set_ylabel("PR-AUC")
fig.tight_layout(); fig.savefig("paper/fig_tradeoff.pdf")
print("figs ok")
