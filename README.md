# Predicting Cancer Therapy-Related Cardiac Dysfunction in HER2-Positive Breast Cancer

Code repository for the manuscript:

**"A parsimonious Cox proportional hazards model outperforms the HFA-ICOS risk score for prediction of cancer therapy-related cardiac dysfunction in HER2-positive breast cancer"**

---

## Overview

This repository contains all analysis code for a study developing and validating a four-variable Cox proportional hazards model for pre-treatment prediction of CTRCD in women with HER2-positive breast cancer receiving potentially cardiotoxic chemotherapy.

The model uses four routinely available variables: age, resting heart rate, baseline LVEF, and prior anthracycline exposure. It was benchmarked against the HFA-ICOS cardiotoxicity risk score using C-index, decision curve analysis, and net reclassification improvement.

---

## Data

Data are from the publicly available **BC_cardiotox dataset**:

> Minchole A, Camps J, Cedilnik N, et al. BC_cardiotox: A cardiotoxicity dataset for breast cancer patients. *Sci Data*. 2023;10:542. doi:10.1038/s41597-023-02419-1

Download from Figshare: https://doi.org/10.6084/m9.figshare.22650748

Place the downloaded files in a `data/` directory in the project root before running any scripts.

---

## Scripts

Run in order from the project root:

| Script | Description |
|--------|-------------|
| `01_eda.py` | Exploratory data analysis |
| `02_baseline_cox.py` | HFA-ICOS benchmark + baseline Cox model |
| `02b_sensitivity_analysis.py` | Sensitivity analyses |
| `03_random_survival_forest.py` | Random survival forest comparison |
| `04_tdi_waveform_model.py` | TDI waveform feature extraction |
| `05_fusion_model.py` | Fusion model (tabular + waveform) |
| `06_calibration_and_decision_curves.py` | Calibration + decision curve analysis (Figure 2) |
| `07_model_refinement.py` | Recalibration, bootstrap CIs, NRI, e' analysis (Figure 5) |
| `08_competing_risks.py` | Fine-Gray competing risks analysis (Figure 4) |
| `09_subgroup_analyses.py` | Subgroup analyses (Figure 3) |
| `10_temporal_validation.py` | Temporal split validation |
| `figure1_km_tertiles.py` | KM cumulative incidence by risk tertile (Figure 1) |

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `lifelines`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `scipy`

---

## Results

Figures are saved to `results/`. Key outputs:

- Cross-validated C-index: **0.743** vs HFA-ICOS 0.663
- Bootstrap C-index: **0.831** (95% CI 0.756–0.895)
- NRI vs HFA-ICOS: **+0.625**
- Low-risk tertile 5-year CTRCD incidence: **2.3%** (no new events after year 2)
- High-risk tertile 5-year CTRCD incidence: **26.5%**

---

## License

MIT

