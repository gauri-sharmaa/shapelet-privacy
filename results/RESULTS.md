## Detection (test accounts) and privacy leakage

| Model | PR-AUC | ROC-AUC | Recall@1%FPR | MIA AUC | Re-ID unique | Re-ID top-1 |
|---|---|---|---|---|---|---|
| catboost | 0.984 ± 0.001 | 0.996 ± 0.000 | 0.857 ± 0.028 | 0.503 ± 0.002 | n/a | n/a |
| minirocket | 0.956 ± 0.001 | 0.988 ± 0.000 | 0.666 ± 0.018 | 0.506 ± 0.000 | n/a | n/a |
| shapelet transform (STC) | 0.869 | 0.943 | 0.447 | 0.497 | 0.170 | 0.215 |
| dilated shapelets (RDST) | 0.952 | 0.987 | 0.618 | 0.521 | 0.834 | – |

## RDST reconstruction attack

| Release | PR-AUC | Shapelets singling out one customer | Customers singled out | of which suspicious | Non-zero cells revealed (median) |
|---|---|---|---|---|---|
| RDST (values + mean/std) | 0.952 | 0.834 | 803.000 | – | 12.000 |

## Defenses (STC shapelets)

| Release | PR-AUC | Re-ID unique | Re-ID top-1 | MIA AUC |
|---|---|---|---|---|
| none | 0.869 | 0.170 | 0.215 | 0.497 |
| noise-0.1 | 0.871 | 0.000 | 0.140 | 0.498 |
| noise-0.25 | 0.866 | 0.000 | 0.035 | 0.500 |
| noise-0.5 | 0.869 | 0.000 | 0.005 | 0.499 |
| noise-1.0 | 0.865 | 0.000 | 0.000 | 0.497 |
| kanon-5 | 0.868 | 0.005 | 0.140 | 0.497 |
| kanon-20 | 0.869 | 0.005 | 0.155 | 0.497 |
| kanon-50 | 0.868 | 0.000 | 0.125 | 0.497 |
| learned | 0.957 | – | – | 0.501 |
