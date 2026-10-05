"""Aggregate results/*.json across seeds into tables (markdown) and figures."""
import glob
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def g(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def main():
    out = []
    base = load("results/baselines_seed*.json") or load("results/baselines.json")
    stc = load("results/stc_attack_seed*.json")
    rdst = load("results/rdst_attack_seed*.json")
    rdst_n = load("results/rdst_normonly_attack_seed*.json")
    dfn = load("results/defenses_seed*.json")

    out.append("## Detection (test accounts) and privacy leakage\n")
    out.append("| Model | PR-AUC | ROC-AUC | Recall@1%FPR | MIA AUC | Re-ID unique | Re-ID top-1 |")
    out.append("|---|---|---|---|---|---|---|")
    for name in ["catboost", "minirocket"]:
        out.append(f"| {name} | {ms([g(b, name, 'pr_auc') for b in base])} | {ms([g(b, name, 'roc_auc') for b in base])} | "
                   f"{ms([g(b, name, 'recall@1%fpr') for b in base])} | {ms([g(b, name, 'mia_conf', 'auc') for b in base])} | n/a | n/a |")
    out.append(f"| shapelet transform (STC) | {ms([g(r, 'detection', 'pr_auc') for r in stc])} | {ms([g(r, 'detection', 'roc_auc') for r in stc])} | "
               f"{ms([g(r, 'detection', 'recall@1%fpr') for r in stc])} | {ms([g(r, 'mia', 'auc') for r in stc])} | "
               f"{ms([g(r, 'reid', 'unique_match') for r in stc])} | {ms([g(r, 'reid', 'top1') for r in stc])} |")
    out.append(f"| dilated shapelets (RDST) | {ms([g(r, 'detection', 'pr_auc') for r in rdst])} | {ms([g(r, 'detection', 'roc_auc') for r in rdst])} | "
               f"{ms([g(r, 'detection', 'recall@1%fpr') for r in rdst])} | {ms([g(r, 'mia', 'auc') for r in rdst])} | "
               f"{ms([g(r, 'reid', 'frac_unique_customer') for r in rdst])} | – |")

    if rdst:
        out.append("\n## RDST reconstruction attack\n")
        out.append("| Release | PR-AUC | Shapelets singling out one customer | Customers singled out | of which suspicious | Non-zero cells revealed (median) |")
        out.append("|---|---|---|---|---|---|")
        for name, rs in [("RDST (values + mean/std)", rdst), ("RDST normalised-only, stats withheld", rdst_n)]:
            if not rs:
                continue
            out.append(f"| {name} | {ms([g(r, 'detection', 'pr_auc') for r in rs])} | {ms([g(r, 'reid', 'frac_unique_customer') for r in rs])} | "
                       f"{ms([g(r, 'reid', 'n_distinct_customers_singled_out') for r in rs])} | {ms([g(r, 'reid', 'n_singled_out_suspicious') for r in rs])} | "
                       f"{ms([g(r, 'attribute', 'median_nonzero_cells_revealed_per_unique_shapelet') for r in rs])} |")

    if dfn:
        out.append("\n## Defenses (STC shapelets)\n")
        out.append("| Release | PR-AUC | Re-ID unique | Re-ID top-1 | MIA AUC |")
        out.append("|---|---|---|---|---|")
        keys = list(dfn[0].keys())
        pts = []
        for k in keys:
            pr = [g(r, k, "detection", "pr_auc") for r in dfn]
            un = [g(r, k, "reid", "unique_match") for r in dfn]
            t1 = [g(r, k, "reid", "top1") for r in dfn]
            out.append(f"| {k} | {ms(pr)} | {ms(un)} | {ms(t1)} | {ms([g(r, k, 'mia', 'auc') for r in dfn])} |")
            if un[0] is not None:
                pts.append((k, np.mean(pr), np.mean(un)))
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for k, pr, un in pts:
            ax.scatter(un, pr, s=30)
            ax.annotate(k, (un, pr), fontsize=7, xytext=(3, 3), textcoords="offset points")
        cb = [g(b, "catboost", "pr_auc") for b in base]
        if cb[0] is not None:
            ax.axhline(np.mean(cb), ls="--", lw=1, color="gray")
            ax.text(0.0, np.mean(cb), " CatBoost", fontsize=7, va="bottom")
        ax.set_xlabel("Shapelets uniquely re-identifying a customer")
        ax.set_ylabel("Detection PR-AUC")
        fig.tight_layout()
        fig.savefig("results/privacy_utility.png", dpi=200)

    sweep = load("results/sweep_k*.json")
    if sweep:
        out.append("\n## Leakage vs number of released shapelets\n")
        out.append("| k | PR-AUC | Re-ID unique | # customers singled out | of which suspicious |")
        out.append("|---|---|---|---|---|")
        for r in sorted(sweep, key=lambda r: r["k"]):
            out.append(f"| {r['k']} | {r['detection']['pr_auc']:.3f} | {r['reid']['unique_match']:.3f} | "
                       f"{r['n_unique_customers']} | {r['n_unique_suspicious']} |")

    open("results/RESULTS.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
