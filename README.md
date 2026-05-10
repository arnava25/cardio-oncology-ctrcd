# Predicting Cardiac Dysfunction in HER2-Positive Breast Cancer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## The Problem

Survival rates for HER2-positive breast cancer have improved dramatically over the past two decades. But the chemotherapy regimens responsible for those gains, particularly trastuzumab and anthracyclines, carry a significant risk of cancer therapy-related cardiac dysfunction (CTRCD): measurable decline in heart function that can force treatment interruption, require lifelong cardiac monitoring, and in some cases become permanent.

Identifying which patients are at high cardiac risk before treatment begins would allow oncologists and cardiologists to personalize surveillance, adjust regimens, or intervene early. The challenge is doing this accurately using information that is routinely available at baseline, without requiring specialized imaging or complex scoring systems.

Existing tools like the HFA-ICOS risk score attempt this, but their clinical utility has not been rigorously benchmarked against modern survival modeling approaches.

This project does that benchmarking, and the results favor a simpler model.

## What This Is

Code and results for the manuscript:

**"A parsimonious Cox proportional hazards model outperforms the HFA-ICOS risk score for prediction of cancer therapy-related cardiac dysfunction in HER2-positive breast cancer"**

The study develops and validates a four-variable Cox proportional hazards model for pre-treatment CTRCD prediction, benchmarks it against the HFA-ICOS score using C-index, decision curve analysis, and net reclassification improvement, and examines performance across subgroups and temporal splits.

## Key Results

| Metric | Four-Variable Cox Model | HFA-ICOS Score |
|---|---|---|
| Cross-validated C-index | **0.743** | 0.663 |
| Bootstrap C-index (95% CI) | **0.831** (0.756 to 0.895) | |
| NRI vs HFA-ICOS | **+0.625** | reference |

Risk stratification by tertile:

| Risk Group | 5-Year CTRCD Incidence |
|---|---|
| Low | 2.3% (no new events after year 2) |
| High | 26.5% |

The model uses four variables available at any standard pre-treatment workup: age, resting heart rate, baseline left ventricular ejection fraction (LVEF), and prior anthracycline exposure. No specialized imaging or proprietary scoring tools required.

## Why a Simpler Model Winning Matters

The HFA-ICOS score incorporates many variables and was developed by expert consensus. A four-variable Cox model outperforming it on discrimination, calibration, decision curve analysis, and net reclassification is a clinically meaningful finding: it suggests that a model trained directly on outcome data, even with minimal predictors, can extract more prognostic signal than a manually weighted checklist.

This has practical implications. A model this parsimonious is easier to implement in electronic health records, easier to explain to patients, and more likely to generalize across clinical settings.

## Data

Data are from the publicly available BC_cardiotox dataset:

> Minchole A, Camps J, Cedilnik N, et al. BC_cardiotox: A cardiotoxicity dataset for breast cancer patients. *Sci Data*. 2023;10:542. doi:10.1038/s41597-023-02419-1

Download from Figshare: https://doi.org/10.6084/m9.figshare.22650748

Place downloaded files in a `data/` directory at the project root before running any scripts.

## Installation

```bash
git clone https://github.com/arnava25/cardio-oncology-ctrcd.git
cd cardio-oncology-ctrcd
pip install -r requirements.txt
```

Key dependencies: `lifelines`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `scipy`

## Scripts

Run in order from the project root:

| Script | Description |
|---|---|
| `01_eda.py` | Exploratory data analysis |
| `02_baseline_cox.py` | HFA-ICOS benchmark and baseline Cox model |
| `02b_sensitivity_analysis.py` | Sensitivity analyses |
| `03_random_survival_forest.py` | Random survival forest comparison |
| `04_tdi_waveform_model.py` | TDI waveform feature extraction |
| `05_fusion_model.py` | Fusion model (tabular + waveform) |
| `06_calibration_and_decision_curves.py` | Calibration and decision curve analysis (Figure 2) |
| `07_model_refinement.py` | Recalibration, bootstrap CIs, NRI, e analysis (Figure 5) |
| `08_competing_risks.py` | Fine-Gray competing risks analysis (Figure 4) |
| `09_subgroup_analyses.py` | Subgroup analyses (Figure 3) |
| `10_temporal_validation.py` | Temporal split validation |
| `figure1_km_tertiles.py` | Kaplan-Meier cumulative incidence by risk tertile (Figure 1) |

## Output

Figures are saved to `results/`. The full analysis pipeline reproduces all manuscript figures and tables.

## Repository Structure

```
cardio-oncology-ctrcd/
├── scripts/
│   ├── 01_eda.py
│   ├── 02_baseline_cox.py
│   ├── 02b_sensitivity_analysis.py
│   ├── 03_random_survival_forest.py
│   ├── 04_tdi_waveform_model.py
│   ├── 05_fusion_model.py
│   ├── 06_calibration_and_decision_curves.py
│   ├── 07_model_refinement.py
│   ├── 08_competing_risks.py
│   ├── 09_subgroup_analyses.py
│   ├── 10_temporal_validation.py
│   └── figure1_km_tertiles.py
├── results/
├── data/
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Questions, collaborations, or feedback welcome. Open an issue or reach out via GitHub.
