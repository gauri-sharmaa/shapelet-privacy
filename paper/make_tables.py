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
\caption{SAML-D: detection on held-out accounts and leakage of the released model (mean $\pm$ std over %d seeds). Re-ID: share of released shapelets that match \emph{only} their source customer among %s candidate accounts.}
\label{tab:main}
\small\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%%
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
t.append(f"RDST, norm.-only, no stats & {ms([g(r, 'detection', 'pr_auc') for r in rdn])} & {ms([g(r, 'detection', 'recall@1%fpr') for r in rdn])} & "
         f"{ms([g(r, 'mia', 'auc') for r in rdn])} & {ms([g(r, 'reid', 'frac_unique_customer') for r in rdn], True)} \\\\")
t.append(f"Learned shapelets & {ms([g(r, 'learned', 'detection', 'pr_auc') for r in dfn])} & {ms([g(r, 'learned', 'detection', 'recall@1%fpr') for r in dfn])} & "
         f"{ms([g(r, 'learned', 'mia', 'auc') for r in dfn])} & 0.0 \\\\")
t.append(r"\bottomrule\end{tabular}}\end{table}")

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


# ---- Realistic adversary (SAML-D) ----
real = load("realistic_seed*.json")
def rv(model, key, field):
    return [r[model][key][field] for r in real if model in r and key in r[model]]
t2 = [r"""\begin{table}[t]
\caption{Re-identification by a realistic adversary (SAML-D, 300 sampled RDST shapelets, mean over seeds). The adversary's copy of the data has perturbed amounts and/or misses transactions, and holds all or half of the accounts; it claims an identity only if the nearest account is at least $2\times$ closer than the runner-up. Cust.: customers correctly re-identified; Prec.: share of claims that are correct.}
\label{tab:real}
\small\setlength{\tabcolsep}{3pt}
\centering
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{All accounts} & \multicolumn{2}{c}{Half of accounts} \\
Adversary's view & Cust. & Prec. & Cust. & Prec. \\
\midrule"""]
for vname, lab in [("exact", "Exact copy"), ("amounts ±5%", r"Amounts $\pm$5\%"),
                   ("80% of txns, ±5%", r"80\% of txns, $\pm$5\%"), ("50% of txns, ±5%", r"50\% of txns, $\pm$5\%")]:
    row = [lab]
    for pf in ["1.0", "0.5"]:
        k = f"{vname}|pool={pf}|ratio=2.0"
        row += [f"{np.mean(rv('rdst', k, 'customers_exposed')):.0f}", f"{100*np.mean(rv('rdst', k, 'precision')):.0f}\\%"]
    t2.append(" & ".join(row) + r" \\")
t2.append(r"\bottomrule\end{tabular}\end{table}")

# ---- Second benchmark and real data (AMLworld, Berka) ----
def la(ds, p):
    return [json.load(open(f)) for f in sorted(glob.glob(f"runs/{ds}/results/" + p))]
ab, ast, ard, arn, adf = (la("amlworld", p) for p in ["baselines_seed*.json", "stc_attack_seed*.json", "rdst_attack_seed*.json", "rdst_normonly_attack_seed*.json", "defenses_seed*.json"])
bb, bst, brd, brn, bdf = (la("berka", p) for p in ["baselines_seed*.json", "stc_attack_seed*.json", "rdst_attack_seed*.json", "rdst_normonly_attack_seed*.json", "defenses_seed*.json"])
breal = la("berka", "realistic_seed*.json")
t3 = [r"""\begin{table}[t]
\caption{AMLworld HI-Small (23,409 accounts, 6-hour bins) and Berka (4,500 real accounts, monthly bins, proxy task); mean over seeds. MIA: aggregate membership AUC. Re-ID: share of shapelets matching only their source; Cust.: customers singled out per model.}
\label{tab:aml}
\small\setlength{\tabcolsep}{2.5pt}
\centering
\begin{tabular}{lcccc}
\toprule
Model & PR-AUC & MIA AUC & Re-ID (\%) & Cust. \\
\midrule"""]
for title, B, S, Rd, Rn, Df in [("AMLworld", ab, ast, ard, arn, adf), ("Berka (real)", bb, bst, brd, brn, bdf)]:
    t3.append(r"\multicolumn{5}{l}{\emph{" + title + r"}} \\")
    for n, lab in [("catboost", "CatBoost"), ("minirocket", "MiniRocket")]:
        t3.append(f"{lab} & {ms([g(b, n, 'pr_auc') for b in B])} & {ms([g(b, n, 'mia_conf', 'auc') for b in B])} & n/a & n/a \\\\")
    t3.append(f"Shapelet transform & {ms([g(r, 'detection', 'pr_auc') for r in S])} & {ms([g(r, 'mia', 'auc') for r in S])} & "
              f"{ms([g(r, 'reid', 'unique_match') for r in S], True)} & -- \\\\")
    t3.append(f"RDST & {ms([g(r, 'detection', 'pr_auc') for r in Rd])} & {ms([g(r, 'mia', 'auc') for r in Rd])} & "
              f"{ms([g(r, 'reid', 'frac_unique_customer') for r in Rd], True)} & {ms([g(r, 'reid', 'n_distinct_customers_singled_out') for r in Rd], dig=0)} \\\\")
    if Rn:
        t3.append(f"RDST, norm.-only & {ms([g(r, 'detection', 'pr_auc') for r in Rn])} & {ms([g(r, 'mia', 'auc') for r in Rn])} & "
                  f"{ms([g(r, 'reid', 'frac_unique_customer') for r in Rn], True)} & {ms([g(r, 'reid', 'n_distinct_customers_singled_out') for r in Rn], dig=0)} \\\\")
    if Df and "learned" in Df[0]:
        t3.append(f"Learned shapelets & {ms([g(r, 'learned', 'detection', 'pr_auc') for r in Df])} & {ms([g(r, 'learned', 'mia', 'auc') for r in Df])} & 0.0 & 0 \\\\")
t3.append(r"\bottomrule\end{tabular}\end{table}")
open("paper/tables.tex", "a").write("\n".join(t2 + t3) + "\n")

probe = load("learned_probe_seed*.json")

if probe:
    P = probe[0]
    t4 = [r"""\begin{table}[t]
\caption{Pushing learned shapelets to memorise (SAML-D, seed 0). Nearest-is-member: share of shapelets whose nearest account is a training member, vs.\ the members' share of the pool (chance). Claims: confident re-identifications ($d_2 \geq 2 d_1$).}
\label{tab:probe}
\small\setlength{\tabcolsep}{3pt}
\centering
\begin{tabular}{lccc}
\toprule
Training & PR-AUC & Nearest-is-member (chance) & Claims \\
\midrule"""]
    for k, v in P.items():
        t4.append(f"{k} & {v['detection']['pr_auc']:.3f} & {100*v['nearest_is_member']:.0f}\\% ({100*v['member_base_rate']:.0f}\\%) & {v['confident_claims']} \\\\")
    t4.append(r"\bottomrule\end{tabular}\end{table}")
    open("paper/tables.tex", "a").write("\n".join(t4) + "\n")

# ---- Beyond shapelets (prototype networks) and DP-SGD learned shapelets ----
def lj(d, p):
    return [json.load(open(f)) for f in sorted(glob.glob(d + p))]
DS = [("SAML-D", "results/"), ("AMLworld", "runs/amlworld/results/"), ("Berka", "runs/berka/results/")]
t5 = [r"""\begin{table}[t]
\caption{Beyond shapelets and a formal defence (mean over 3 seeds). ProtoNet: prototype network whose prototypes are projected onto training patches (ProtoPNet-style); Re-ID: projected prototypes matching only their source customer. DP: learned shapelets trained with DP-SGD ($\delta = 1/n$); PR-AUC at $\varepsilon=\infty$ / 8 / 1.}
\label{tab:proto}
\small\setlength{\tabcolsep}{2.5pt}
\centering
\begin{tabular}{lccc}
\toprule
& SAML-D & AMLworld & Berka \\
\midrule"""]
P = {n: lj(d, "protonet_seed*.json") for n, d in DS}
Dp = {n: lj(d, "dp_shapelets_seed*.json") for n, d in DS}
row = lambda lab, f: t5.append(lab + " & " + " & ".join(f(n) for n, _ in DS) + r" \\")
row("ProtoNet PR-AUC", lambda n: f"{np.mean([x['protonet']['pr_auc_after_projection'] for x in P[n]]):.3f}")
row("ProtoNet Re-ID (\\%)", lambda n: f"{100*np.mean([x['protonet']['frac_correct_source'] for x in P[n]]):.0f}")
row("Customers exposed / 20", lambda n: f"{np.mean([x['protonet']['n_customers_exposed'] for x in P[n]]):.1f}")
t5.append(r"\midrule")
for e, lab in [("inf", r"DP $\varepsilon=\infty$"), ("8.0", r"DP $\varepsilon=8$"), ("1.0", r"DP $\varepsilon=1$")]:
    row(lab + " PR-AUC", lambda n, e=e: f"{np.mean([x[e]['detection']['pr_auc'] for x in Dp[n]]):.3f}")
t5.append(r"\bottomrule\end{tabular}\end{table}")
open("paper/tables.tex", "a").write("\n".join(t5) + "\n")
N_extra = {
 "protoReid": f"{100*np.mean([x['protonet']['frac_correct_source'] for n,_ in DS for x in P[n]]):.0f}",
 "protoReidMin": f"{100*min(np.mean([x['protonet']['frac_correct_source'] for x in P[n]]) for n,_ in DS):.0f}",
 "protoReidMax": f"{100*max(np.mean([x['protonet']['frac_correct_source'] for x in P[n]]) for n,_ in DS):.0f}",
 "dpSamldOne": f"{np.mean([x['1.0']['detection']['pr_auc'] for x in Dp['SAML-D']]):.3f}",
 "dpSamldInf": f"{np.mean([x['inf']['detection']['pr_auc'] for x in Dp['SAML-D']]):.3f}",
 "dpBerkaOne": f"{np.mean([x['1.0']['detection']['pr_auc'] for x in Dp['Berka']]):.2f}",
 "dpBerkaInf": f"{np.mean([x['inf']['detection']['pr_auc'] for x in Dp['Berka']]):.2f}",
 "dpNearSamld": f"{100*np.mean([x['1.0']['nearest_is_member'] for x in Dp['SAML-D']]):.0f}",
}

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
    "realCust": f"{np.mean(rv('rdst', 'amounts ±5%|pool=1.0|ratio=2.0', 'customers_exposed')):.0f}",
    "realPrec": f"{100*np.mean(rv('rdst', 'amounts ±5%|pool=1.0|ratio=2.0', 'precision')):.0f}",
    "realSusp": f"{np.mean(rv('rdst', 'amounts ±5%|pool=1.0|ratio=2.0', 'suspicious_exposed')):.0f}",
    "realEightyCust": f"{np.mean(rv('rdst', '80% of txns, ±5%|pool=1.0|ratio=2.0', 'customers_exposed')):.0f}",
    "realEightyPrec": f"{100*np.mean(rv('rdst', '80% of txns, ±5%|pool=1.0|ratio=2.0', 'precision')):.0f}",
    "realHalfPrec": f"{100*np.mean(rv('rdst', '50% of txns, ±5%|pool=1.0|ratio=2.0', 'precision')):.0f}",
    "realNormCust": f"{np.mean(rv('rdst_normonly', 'amounts ±5%|pool=1.0|ratio=2.0', 'customers_exposed')):.0f}",
    "realNormPrec": f"{100*np.mean(rv('rdst_normonly', 'amounts ±5%|pool=1.0|ratio=2.0', 'precision')):.0f}",
    "amlRdstUnique": f"{mean([g(r, 'reid', 'frac_unique_customer') for r in ard], 100):.0f}",
    "amlRdstCust": f"{mean([g(r, 'reid', 'n_distinct_customers_singled_out') for r in ard]):.0f}",
    "amlRdstSusp": f"{mean([g(r, 'reid', 'n_singled_out_suspicious') for r in ard]):.0f}",
    "amlRdstPR": f"{mean([g(r, 'detection', 'pr_auc') for r in ard]):.2f}",
    "amlCbPR": f"{mean([g(b, 'catboost', 'pr_auc') for b in ab]):.2f}",
    "amlCbMIA": f"{mean([g(b, 'catboost', 'mia_conf', 'auc') for b in ab]):.2f}",
    "amlStcUnique": f"{mean([g(r, 'reid', 'unique_match') for r in ast], 100):.0f}",
    "probeNear": f"{100*mean([g(p, k, 'nearest_is_member') for p in probe for k in p]):.0f}" if probe else "--",
    "probeConf": f"{max([g(p, k, 'confident_claims_on_members') for p in probe for k in p]):.0f}" if probe else "--",
}
ext = load("rdst_extra_seed*.json")
scale = [e["scale"] for e in ext if "scale" in e]
if scale:
    keys = sorted(scale[0], key=int)
    N.update({"scaleN": f"{int(keys[-1]):,}", "scaleExtra": f"{int(keys[-1]) - int(keys[0]):,}",
              "scaleKeep": f"{100 * scale[0][keys[-1]]:.0f}"})
if ext:
    N.update({"auditTpr": f"{100 * mean([e['audit']['tpr@0.1%fpr'] for e in ext]):.1f}",
              "auditMembers": f"{mean([e['audit']['members_singled_out'] for e in ext]):.0f}",
              "auditNonmembers": f"{mean([e['audit']['nonmembers_singled_out'] for e in ext]):.0f}"})
def brv(model, key, field):
    return [r[model][key][field] for r in breal if model in r and key in r[model]]
N.update({
    "berkaUnique": f"{mean([g(r, 'reid', 'frac_unique_customer') for r in brd], 100):.0f}",
    "berkaCust": f"{mean([g(r, 'reid', 'n_distinct_customers_singled_out') for r in brd]):.0f}",
    "berkaStcUnique": f"{mean([g(r, 'reid', 'unique_match') for r in bst], 100):.0f}",
    "berkaRdstPR": f"{mean([g(r, 'detection', 'pr_auc') for r in brd]):.2f}",
    "berkaLearnPR": f"{mean([g(r, 'learned', 'detection', 'pr_auc') for r in bdf]):.2f}",
    "berkaRealCust": f"{mean(brv('rdst', 'amounts ±5%|pool=1.0|ratio=2.0', 'customers_exposed')):.0f}" if breal else "--",
    "berkaRealPrec": f"{mean(brv('rdst', 'amounts ±5%|pool=1.0|ratio=2.0', 'precision'), 100):.0f}" if breal else "--",
})
for k in ["scaleN", "scaleExtra", "scaleKeep", "auditTpr", "auditMembers", "auditNonmembers"]:
    N.setdefault(k, "--")
N.update(N_extra)
import json as _json, os as _os
_lf = "runs/berka/results/lira_final.json"
if _os.path.exists(_lf):
    L = _json.load(open(_lf))
    N.update({"liraShadows": str(L["n_shadows"]),
              "liraRdstAuc": f"{L['rdst_lira']['auc']:.2f}", "liraRdstTpr": f"{100*L['rdst_lira']['tpr@0.1%fpr']:.1f}",
              "liraAuditTpr": f"{100*L['rdst_per_explanation']['tpr@0.1%fpr']:.1f}",
              "liraCbAuc": f"{L['catboost_lira']['auc']:.2f}", "liraCbTpr": f"{100*L['catboost_lira']['tpr@0.1%fpr']:.1f}"})
open("paper/numbers.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in N.items()) + "\n")
print(N)
