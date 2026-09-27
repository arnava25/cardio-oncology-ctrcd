# Cardio-Oncology CTRCD Prediction Model

A parsimonious four-variable Cox proportional hazards model for pre-treatment prediction of cancer therapy-related cardiac dysfunction (CTRCD) in HER2-positive breast cancer, developed and internally validated on the publicly available BC_cardiotox dataset (n=531).

The model uses four routinely available pre-treatment variables — age, resting heart rate, baseline LVEF, and prior anthracycline exposure — and is benchmarked against the HFA-ICOS risk score using discrimination (C-index), calibration, decision curve analysis, and net reclassification improvement. It is a **development study with internal (bootstrap) validation**; external validation has not been performed and is required before any clinical use.

## Key results

| Metric | Cox model | HFA-ICOS |
|---|---|---|
| Cross-validated C-index | 0.714 | — |
| Optimism-corrected C-index (95% CI) | **0.723** (0.705 to 0.742) | — |
| Paired C-index on identical subset (n=477, 46 events) | 0.726 | 0.649 |
| Paired difference (95% CI) | **+0.077** (-0.006 to +0.162) | |
| NRI vs HFA-ICOS | +0.625 | |

Calibration required recalibration: calibration slope 1.52, intercept +1.75; expected/observed ratio 1.00 after in-sample Platt recalibration at the 2-year landmark (n=208, 33 events). The improvement over HFA-ICOS is modest and its confidence interval approaches zero; the model is best read as competitive with, and likely superior to, the consensus score, pending external validation.

## Reproducing the analysis

All reported numbers are reproduced from the raw data by a single script:

```
python analysis.py          # cohort, Table 2, discrimination, CV-lambda, paired comparison, calibration, bake-off, tertiles, Brier
python figures.py           # Figures 1-3 (KM tertiles, calibration/DCA, subgroups)
python supp.py              # Supplementary Figure 1 (bootstrap validation) + Central Illustration
```

`analysis.py` is the single source of truth: it fits the model and computes every reported metric deterministically (seed = 42). The scripts in `scripts/` are retained exploratory analyses (EDA) and are not required to reproduce the manuscript results.

## Data

The BC_cardiotox dataset is publicly available at Figshare (DOI: 10.6084/m9.figshare.22650748). Place `BC_cardiotox_clinical_variables.csv` under `data/` (semicolon-separated, comma decimal). The dataset is not redistributed in this repository.

## Model

Cox proportional hazards, four predictors, ridge penalty (λ=0.1; cross-validation-selected λ=0.05 gives materially unchanged performance). Coefficients (per unit): age HR 1.019, resting heart rate HR 1.018, baseline LVEF HR 0.978, prior anthracycline exposure HR 1.778.

## Citation

Amit A. A Parsimonious Cox Model Outperforms HFA-ICOS for Cardiotoxicity Prediction in HER2-Positive Breast Cancer. (Under review.)