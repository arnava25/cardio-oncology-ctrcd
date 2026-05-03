"""
03_random_survival_forest.py — Random Survival Forest
Cardio-oncology CTRCD prediction project

Captures interaction effects that the linear Cox model misses.
This is the model you run AFTER establishing the Cox baseline.

Run from the project root:
    python scripts/03_random_survival_forest.py

Dependencies (install once):
    pip install scikit-survival

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

# ── try importing scikit-survival ──────────────────────────────────────────
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print("scikit-survival not installed. Run:")
    print("    pip install scikit-survival")
    print("Then re-run this script.")
    exit(1)

# ── paths ──────────────────────────────────────────────────────────────────
DATA = Path("data/BC_cardiotox_clinical_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, sep=";", decimal=",")
print(f"Loaded {df.shape[0]} patients, events={df['CTRCD'].sum()}")

# ── features ───────────────────────────────────────────────────────────────
# Using the same parsimonious set as Cox for fair comparison
# RSF can handle more variables without overfitting as badly,
# but we keep it matched so the comparison is apples-to-apples.
FEATURES = ["age", "heart_rate", "LVEF", "DL", "AC", "antiHER2"]

df_model = df[FEATURES + ["CTRCD", "time"]].dropna()
print(f"Complete cases: {len(df_model)}, events: {df_model['CTRCD'].sum()}")

# scikit-survival needs a structured array for the outcome
y = Surv.from_dataframe("CTRCD", "time", df_model)
X = df_model[FEATURES].values

# ── fit RSF ────────────────────────────────────────────────────────────────
rsf = RandomSurvivalForest(
    n_estimators=300,
    min_samples_split=10,  # prevents overfitting on small event count
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

# ── cross-validated C-index ────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, df_model["CTRCD"])):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rsf_cv = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rsf_cv.fit(X_train, y_train)

    # predict_cumulative_hazard_function → higher = higher risk
    pred = rsf_cv.predict(X_test)
    c, _, _, _, _ = concordance_index_censored(y_test["CTRCD"], y_test["time"], pred)
    cv_scores.append(c)
    print(f"  Fold {fold+1}: C-index = {c:.3f}")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"\nRSF CV C-index: {cv_mean:.3f} ± {cv_std:.3f}")
print(f"Cox CV C-index (from script 02): ~0.712 ± 0.113")
print(f"HFA-ICOS baseline:                0.663")

# ── fit on full data for feature importance ────────────────────────────────
rsf.fit(X, y)

# Variable importance (permutation-based: how much does C-index drop
# when we shuffle each variable?)
from sklearn.inspection import permutation_importance


# Wrap RSF predict for sklearn permutation_importance
# We need a scorer that returns a scalar (C-index)
class RSFScorer:
    def __init__(self, model, y):
        self.model = model
        self.y = y

    def __call__(self, estimator, X, y_ignored):
        pred = estimator.predict(X)
        c, _, _, _, _ = concordance_index_censored(
            self.y["CTRCD"], self.y["time"], pred
        )
        return c


print("\n=== Variable importance (permutation) ===")
scorer = RSFScorer(rsf, y)
baseline_c = scorer(rsf, X, None)
importances = []
rng = np.random.default_rng(42)
for i, feat in enumerate(FEATURES):
    scores = []
    for _ in range(20):  # 20 permutations per feature
        X_perm = X.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        scores.append(scorer(rsf, X_perm, None))
    drop = baseline_c - np.mean(scores)
    importances.append((feat, drop))
    print(f"  {feat:12s}: importance = {drop:.4f}")

importances.sort(key=lambda x: x[1], reverse=True)

# ── plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#fafaf8")

# Feature importance bar chart
ax = axes[0]
feats, imps = zip(*importances)
colors = ["#1D9E75" if i > 0 else "#D85A30" for i in imps]
ax.barh(feats, imps, color=colors, alpha=0.85)
ax.axvline(0, color="#888780", linewidth=0.8)
ax.set_xlabel("C-index drop when variable shuffled\n(higher = more important)")
ax.set_title("RSF variable importance", fontweight="500")
ax.spines[["top", "right"]].set_visible(False)

# Performance comparison bar chart
ax = axes[1]
model_names = ["HFA-ICOS\n(baseline)", "Cox\n(parsimonious)", "RSF"]
model_means = [0.663, 0.712, cv_mean]
model_errors = [0, 0.113, cv_std]
bar_colors = ["#B4B2A9", "#5DCAA5", "#1D9E75"]
bars = ax.bar(model_names, model_means, color=bar_colors, alpha=0.85, width=0.5)
for i, (mean, err) in enumerate(zip(model_means, model_errors)):
    if err > 0:
        ax.errorbar(
            i, mean, yerr=err, fmt="none", ecolor="#0F6E56", capsize=6, linewidth=1.5
        )
    ax.text(
        i,
        mean + 0.005,
        f"{mean:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="500",
    )
ax.set_ylim(0.5, 0.85)
ax.set_ylabel("C-index / AUC")
ax.set_title("Model comparison", fontweight="500")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
outpath = OUT / "rsf_results.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved plot → {outpath}")
