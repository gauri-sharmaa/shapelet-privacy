"""Generate LaTeX tables (paper/tables.tex) and macro numbers (paper/numbers.tex) from results/*.json."""
import glob
import json
import numpy as np

R = "results/"


def load(p):
    return [json.load(open(f)) for f in sorted(glob.glob(R + p))]


def g(d, *ks):
    for k in ks:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def ms(v, pct=False, dig=3):
    v = [x for x in v if x is not None]
    if not v:
        return "--"
    f = 100 if pct else 1
    d = 1 if pct else dig
    m = np.mean(v) * f
    if len(v) == 1:
        return f"{m:.{d}f}"
    return f"{m:.{d}f}\\,$\\pm$\\,{np.std(v) * f:.{d}f}"


base, stc, rdst = load("baselines_seed*.json"), load("stc_attack_seed*.json"), load("rdst_attack_seed*.json")
rdn, dfn, sw = load("rdst_normonly_attack_seed*.json"), load("defenses_seed*.json"), load("sweep_k*.json")

t = []
t.append(r"""\begin{table}[t]
\caption{Detection on held-out accounts and leakage of the released model (mean $\pm$ std over %d seeds). Re-ID: share of released shapelets that match \emph{only} their source customer among %s candidate accounts.}
\label{tab:main}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Model & PR-AUC & R@1\%%FPR & MIA AUC & Re-ID (\%%) \\
\midrule""" % (len(base), f"{g(stc[0], 'reid', 'n_candidates'):,}" if stc else "?"))
for n, lab in [("catboost", "CatBoost (aggregates)"), ("minirocket", "MiniRocket")]:
    t.append(f"{lab} & {ms([g(b, n, 'pr_auc') for b in base])} & {ms([g(b, n, 'recall@1%fpr') for b in base])} & "
             f"{ms([g(b, n, 'mia_conf', 'auc') for b in base])} & n/a \\\\")
t.append(f"Shapelet transform & {ms([g(r, 'detection', 'pr_auc') for r in stc])} & {ms([g(r, 'detection', 'recall@1%fpr') for r in stc])} & "
         f"{ms([g(r, 'mia', 'auc') for r in stc])} & {ms([g(r, 'reid', 'unique_match') for r in stc], True)} \\\\")
t.append(f"RDST & {ms([g(r, 'detection', 'pr_auc') for r in rdst])} & {ms([g(r, 'detection', 'recall@1%fpr') for r in rdst])} & "
         f"{ms([g(r, 'mia', 'auc') for r in rdst])} & {ms([g(r, 'reid', 'frac_unique_customer') for r in rdst], True)} \\\\")
t.append(f"Learned shapelets & {ms([g(r, 'learned', 'detection', 'pr_auc') for r in dfn])} & {ms([g(r, 'learned', 'detection', 'recall@1%fpr') for r in dfn])} & "
         f"{ms([g(r, 'learned', 'mia', 'auc') for r in dfn])} & 0.0 \\\\")
t.append(r"\bottomrule\end{tabular}\end{table}")

t.append(r"""\begin{table}[t]
\caption{Mitigations (shapelet transform; mean over seeds). Unique: shapelet matches only its source; Top-1: source is the nearest account.}
\label{tab:def}
\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
Release & PR-AUC & Unique (\%) & Top-1 (\%) \\
\midrule""")
names = {"none": "As trained", "noise-0.1": r"Noise $\sigma{=}0.1$", "noise-0.25": r"Noise $\sigma{=}0.25$",
         "noise-0.5": r"Noise $\sigma{=}0.5$", "noise-1.0": r"Noise $\sigma{=}1.0$", "kanon-5": "$k$-anon. mean, $k{=}5$",
         "kanon-20": "$k$-anon. mean, $k{=}20$", "kanon-50": "$k$-anon. mean, $k{=}50$"}
for k, lab in names.items():
    t.append(f"{lab} & {ms([g(r, k, 'detection', 'pr_auc') for r in dfn])} & {ms([g(r, k, 'reid', 'unique_match') for r in dfn], True)} & "
             f"{ms([g(r, k, 'reid', 'top1') for r in dfn], True)} \\\\")
t.append(r"\bottomrule\end{tabular}\end{table}")
open("paper/tables.tex", "w").write("\n".join(t) + "\n")

# Macros used in the text.
def mean(v, f=1):
    v = [x for x in v if x is not None]
    return np.mean(v) * f if v else float("nan")

N = {
    "stcUnique": f"{mean([g(r, 'reid', 'unique_match') for r in stc], 100):.0f}",
    "stcTop": f"{mean([g(r, 'reid', 'top1') for r in stc], 100):.0f}",
    "stcSusp": f"{mean([g(r, 'source_label_counts', 'suspicious') / max(1, g(r, 'n_shapelets')) for r in stc], 100):.0f}",
    "rdstUnique": f"{mean([g(r, 'reid', 'frac_unique_customer') for r in rdst], 100):.0f}",
    "rdstCust": f"{mean([g(r, 'reid', 'n_distinct_customers_singled_out') for r in rdst]):.0f}",
    "rdstSusp": f"{mean([g(r, 'reid', 'n_singled_out_suspicious') for r in rdst]):.0f}",
    "rdstCells": f"{mean([g(r, 'attribute', 'median_nonzero_cells_revealed_per_unique_shapelet') for r in rdst]):.0f}",
    "rdstAmt": f"{mean([g(r, 'attribute', 'median_max_weekly_amount_revealed') for r in rdst]) / 1000:.0f}",
    "rdstPR": f"{mean([g(r, 'detection', 'pr_auc') for r in rdst]):.3f}",
    "rdstMIA": f"{mean([g(r, 'mia', 'auc') for r in rdst]):.2f}",
    "rdnUnique": f"{mean([g(r, 'reid', 'frac_unique_customer') for r in rdn], 100):.0f}",
    "rdnPR": f"{mean([g(r, 'detection', 'pr_auc') for r in rdn]):.3f}",
    "learnPR": f"{mean([g(r, 'learned', 'detection', 'pr_auc') for r in dfn]):.3f}",
    "learnDist": f"{mean([g(r, 'learned', 'nearest_account_dist_median') for r in dfn]):.2f}",
    "stcPR": f"{mean([g(r, 'detection', 'pr_auc') for r in stc]):.3f}",
    "cbPR": f"{mean([g(b, 'catboost', 'pr_auc') for b in base]):.3f}",
    "nSeeds": str(len(base)),
}
open("paper/numbers.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in N.items()) + "\n")
print(N)
