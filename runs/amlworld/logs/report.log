## Detection (test accounts) and privacy leakage

| Model | PR-AUC | ROC-AUC | Recall@1%FPR | MIA AUC | Re-ID unique | Re-ID top-1 |
|---|---|---|---|---|---|---|
| catboost | 0.619 ± 0.004 | 0.803 ± 0.001 | 0.164 ± 0.007 | 0.625 ± 0.004 | n/a | n/a |
| minirocket | 0.475 ± 0.003 | 0.707 ± 0.001 | 0.075 ± 0.004 | 0.586 ± 0.003 | n/a | n/a |
| shapelet transform (STC) | 0.408 ± 0.016 | 0.658 ± 0.015 | 0.062 ± 0.013 | 0.499 ± 0.002 | 0.060 ± 0.011 | 0.067 ± 0.006 |
| dilated shapelets (RDST) | 0.571 ± 0.001 | 0.778 ± 0.001 | 0.140 ± 0.008 | 0.507 ± 0.002 | 0.645 ± 0.012 | – |

## RDST reconstruction attack

| Release | PR-AUC | Shapelets singling out one customer | Customers singled out | of which suspicious | Non-zero cells revealed (median) |
|---|---|---|---|---|---|
| RDST (values + mean/std) | 0.571 ± 0.001 | 0.645 ± 0.012 | 620.667 ± 10.403 | 163.667 ± 14.055 | 10.000 ± 0.000 |
| RDST normalised-only, stats withheld | 0.546 ± 0.005 | 0.404 ± 0.017 | 391.333 ± 14.291 | 110.667 ± 4.028 | 55.000 ± 0.000 |

## Defenses (STC shapelets)

| Release | PR-AUC | Re-ID unique | Re-ID top-1 | MIA AUC |
|---|---|---|---|---|
| none | 0.389 | 0.050 | 0.065 | 0.497 |
| noise-0.1 | 0.400 | 0.000 | 0.010 | 0.496 |
| noise-0.25 | 0.392 | 0.000 | 0.005 | 0.498 |
| noise-0.5 | 0.400 | 0.000 | 0.000 | 0.501 |
| noise-1.0 | 0.429 | 0.000 | 0.000 | 0.502 |
| kanon-5 | 0.390 | 0.010 | 0.035 | 0.494 |
| kanon-20 | 0.390 | 0.000 | 0.005 | 0.494 |
| kanon-50 | 0.387 | 0.000 | 0.010 | 0.494 |
| learned | 0.389 | – | – | 0.500 |
