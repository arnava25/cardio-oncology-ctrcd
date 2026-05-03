"""
02b_sensitivity_analysis.py — Cox sensitivity analysis
Cardio-oncology CTRCD prediction project

Tests whether adding ACprev and RTprev (prior anthracyclines and prior
chest radiation) improves the Cox model. These variables showed the
largest treatment-related gaps in EDA (ACprev: 10% vs 30%, RTprev: 13%
vs 24%) but were excluded from the main model due to ~50 missing values.

Strategy: restrict to complete cases for these variables (~430 patients)
and compare three models head-to-head on the same subset.

Run from project root:
    python scripts/02b_sensitivity_analysis.py

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
DATA = Path('data/BC_cardiotox_clinical_variables.csv')
OUT  = Path('results')
OUT.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, sep=';', decimal=',')
print(f"Full dataset: {len(df)} patients, {df['CTRCD'].sum()} events")

# ── define the three models to compare ────────────────────────────────────
#
#  Model A — original parsimonious Cox (script 02)
#  Model B — adds ACprev and RTprev
#  Model C — adds ACprev, RTprev, and their interaction term
#
# All three evaluated on the SAME restricted subset (complete cases for
# the extended variable set) so comparisons are apples-to-apples.

FEATURES_A = ['age', 'heart_rate', 'LVEF', 'DL', 'AC', 'antiHER2']
FEATURES_B = ['age', 'heart_rate', 'LVEF', 'DL', 'AC', 'antiHER2',
              'ACprev', 'RTprev']

# Restrict to complete cases across all variables needed
all_vars = list(set(FEATURES_B + ['CTRCD', 'time']))
df_restricted = df[all_vars].dropna()
print(f"\nRestricted dataset (complete cases): {len(df_restricted)} patients, "
      f"{df_restricted['CTRCD'].sum()} events")
print(f"Dropped {len(df) - len(df_restricted)} patients due to missing ACprev/RTprev")

# Add interaction term for model C
df_restricted = df_restricted.copy()
df_restricted['ACprev_x_RTprev'] = (
    df_restricted['ACprev'] * df_restricted['RTprev']
)
FEATURES_C = FEATURES_B + ['ACprev_x_RTprev']

# ── cross-validation function ──────────────────────────────────────────────
def cv_cindex(df_data, features, n_splits=5, penalizer=0.05, seed=42):
    """
    Returns (mean_c, std_c) from stratified k-fold cross-validation.
    Also returns per-fold scores for plotting.
    """
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    X      = df_data[features + ['CTRCD', 'time']]

    for train_idx, test_idx in skf.split(X, X['CTRCD']):
        train = X.iloc[train_idx]
        test  = X.iloc[test_idx]
        m = CoxPHFitter(penalizer=penalizer)
        m.fit(train, duration_col='time', event_col='CTRCD')
        c = concordance_index(
            test['time'],
            -m.predict_partial_hazard(test[features]),
            test['CTRCD']
        )
        scores.append(c)

    return np.mean(scores), np.std(scores), scores

# ── run all three models ───────────────────────────────────────────────────
print("\n=== Running 5-fold CV on restricted dataset ===")

print("\nModel A — original parsimonious (age, heart_rate, LVEF, DL, AC, antiHER2)")
mean_a, std_a, folds_a = cv_cindex(df_restricted, FEATURES_A)
print(f"  CV C-index: {mean_a:.3f} ± {std_a:.3f}")

print("\nModel B — adds ACprev + RTprev")
mean_b, std_b, folds_b = cv_cindex(df_restricted, FEATURES_B)
print(f"  CV C-index: {mean_b:.3f} ± {std_b:.3f}")

print("\nModel C — adds ACprev × RTprev interaction")
mean_c, std_c, folds_c = cv_cindex(df_restricted, FEATURES_C)
print(f"  CV C-index: {mean_c:.3f} ± {std_c:.3f}")

# ── fit model B on full restricted data to inspect coefficients ────────────
print("\n=== Model B coefficient table ===")
cph_b = CoxPHFitter(penalizer=0.05)
cph_b.fit(df_restricted[FEATURES_B + ['CTRCD', 'time']],
          duration_col='time', event_col='CTRCD')
cph_b.print_summary(decimals=3,
                    columns=['coef', 'exp(coef)', 'se(coef)', 'p'])

print("\n=== Key new variables ===")
for var in ['ACprev', 'RTprev']:
    hr = cph_b.summary.loc[var, 'exp(coef)']
    p  = cph_b.summary.loc[var, 'p']
    ci_lo = cph_b.summary.loc[var, 'exp(coef) lower 95%']
    ci_hi = cph_b.summary.loc[var, 'exp(coef) upper 95%']
    print(f"  {var}: HR={hr:.2f} (95% CI {ci_lo:.2f}–{ci_hi:.2f}), p={p:.3f}")

# ── decision: which model is best? ────────────────────────────────────────
print("\n=== Summary ===")
print(f"  HFA-ICOS baseline (full dataset):  0.663")
print(f"  Model A — original Cox:            {mean_a:.3f} ± {std_a:.3f}")
print(f"  Model B — + ACprev + RTprev:       {mean_b:.3f} ± {std_b:.3f}")
print(f"  Model C — + interaction:           {mean_c:.3f} ± {std_c:.3f}")

best_mean = max(mean_a, mean_b, mean_c)
if best_mean == mean_b:
    winner = "Model B"
elif best_mean == mean_c:
    winner = "Model C"
else:
    winner = "Model A (original)"
print(f"\n  Best model: {winner} (CV C-index {best_mean:.3f})")
print(f"  Note: with only ~{df_restricted['CTRCD'].sum()} events, differences")
print(f"  within ±0.02 are likely noise — prefer simpler model if tied.")

# ── plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#fafaf8')

# Left: per-fold C-indices for all three models
ax = axes[0]
x   = np.arange(1, 6)
ax.plot(x, folds_a, 'o-', color='#5DCAA5', label='Model A (original)', linewidth=1.5)
ax.plot(x, folds_b, 's-', color='#1D9E75', label='Model B (+ACprev+RTprev)', linewidth=1.5)
ax.plot(x, folds_c, '^-', color='#085041', label='Model C (+interaction)', linewidth=1.5)
ax.axhline(0.663, color='#B4B2A9', linestyle='--', linewidth=1,
           label='HFA-ICOS baseline')
ax.set_xlabel('CV fold')
ax.set_ylabel('C-index')
ax.set_title('Per-fold C-index by model', fontweight='500')
ax.legend(fontsize=9)
ax.set_ylim(0.35, 0.95)
ax.spines[['top', 'right']].set_visible(False)

# Right: mean ± std comparison
ax = axes[1]
model_names = ['HFA-ICOS\n(baseline)', 'Model A\noriginal', 'Model B\n+ACprev\n+RTprev',
               'Model C\n+interaction']
means  = [0.663,  mean_a, mean_b, mean_c]
stds   = [0,      std_a,  std_b,  std_c]
colors = ['#B4B2A9', '#9FE1CB', '#1D9E75', '#085041']
bars   = ax.bar(model_names, means, color=colors, alpha=0.85, width=0.55)
for i, (mean, std) in enumerate(zip(means, stds)):
    if std > 0:
        ax.errorbar(i, mean, yerr=std, fmt='none',
                    ecolor='#444441', capsize=5, linewidth=1.5)
    ax.text(i, mean + 0.006, f'{mean:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='500')
ax.set_ylim(0.50, 0.90)
ax.set_ylabel('CV C-index')
ax.set_title('Model comparison — restricted dataset', fontweight='500')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
outpath = OUT / 'sensitivity_analysis.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#fafaf8')
print(f"\nSaved → {outpath}")