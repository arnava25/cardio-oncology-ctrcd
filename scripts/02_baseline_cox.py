"""
02_baseline_cox.py — HFA-ICOS benchmark + parsimonious Cox survival model
Cardio-oncology CTRCD prediction project

Run from the project root:
    python scripts/02_baseline_cox.py

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA = Path("data/BC_cardiotox_clinical_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, sep=";", decimal=",")
print(f"Loaded {df.shape[0]} patients, events={df['CTRCD'].sum()}")

# ══════════════════════════════════════════════════════════════════════════
# PART 1: HFA-ICOS score — the clinical baseline we need to beat
# Based on Lyon et al. 2020 Eur J Heart Fail
# ══════════════════════════════════════════════════════════════════════════


def hfa_icos_score(row):
    """
    Approximate HFA-ICOS cardiovascular risk score.
    Higher score = higher predicted risk of CTRCD.
    """
    score = 0

    # Treatment factors
    if row["AC"] == 1:
        score += 1  # anthracyclines
    if row["antiHER2"] == 1:
        score += 1  # anti-HER2 therapy
    if row["ACprev"] == 1:
        score += 2  # prior anthracyclines (higher weight)
    if row["antiHER2prev"] == 1:
        score += 1  # prior anti-HER2
    if row["RTprev"] == 1:
        score += 1  # prior chest radiotherapy

    # Age
    if row["age"] >= 80:
        score += 2
    elif row["age"] >= 65:
        score += 1

    # Baseline cardiac function
    if not pd.isna(row["LVEF"]):
        if row["LVEF"] < 50:
            score += 3  # reduced EF
        elif row["LVEF"] < 55:
            score += 2  # mildly reduced

    # Cardiovascular comorbidities
    if row["HTA"] == 1:
        score += 1
    if row["DM"] == 1:
        score += 1
    if row["DL"] == 1:
        score += 1
    if row["CIprev"] == 1:
        score += 2  # prior heart failure
    if row["ICMprev"] == 1:
        score += 2  # prior ischemic cardiomyopathy
    if row["ARRprev"] == 1:
        score += 1
    if row["VALVprev"] == 1:
        score += 1

    return score


def hfa_icos_category(score):
    if score <= 1:
        return "low"
    elif score <= 4:
        return "moderate"
    elif score <= 6:
        return "high"
    else:
        return "very_high"


# Score all patients with complete treatment/comorbidity data
required = [
    "AC",
    "antiHER2",
    "ACprev",
    "antiHER2prev",
    "RTprev",
    "HTA",
    "DL",
    "DM",
    "CIprev",
    "ICMprev",
    "ARRprev",
    "VALVprev",
]
df_scored = df.dropna(subset=required).copy()
df_scored["hfa_score"] = df_scored.apply(hfa_icos_score, axis=1)
df_scored["hfa_category"] = df_scored["hfa_score"].apply(hfa_icos_category)

print(f"\n=== HFA-ICOS scored: {len(df_scored)} patients ===")
print(df_scored["hfa_category"].value_counts())

print("\n=== CTRCD rate by risk category ===")
summary = df_scored.groupby("hfa_category").agg(
    n=("CTRCD", "count"),
    events=("CTRCD", "sum"),
    ctrcd_pct=("CTRCD", lambda x: f"{x.mean()*100:.1f}%"),
)
cat_order = ["low", "moderate", "high", "very_high"]
print(summary.loc[[c for c in cat_order if c in summary.index]])

hfa_auc = roc_auc_score(df_scored["CTRCD"], df_scored["hfa_score"])
print(f"\nHFA-ICOS AUC (baseline to beat): {hfa_auc:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# PART 2: Parsimonious Cox model
# Variables chosen by: statistical significance in full model + clinical prior
#   age        — each year increases hazard (p=0.022)
#   heart_rate — resting HR as autonomic/cardiac stress marker (p=0.005)
#   LVEF       — baseline systolic function (p=0.044)
#   DL         — dyslipidemia as vascular risk (trending signal)
#   AC         — anthracycline exposure
#   antiHER2   — anti-HER2 therapy
# Rule of thumb: need ~10 events per predictor → 47 events → max ~5 predictors
# ══════════════════════════════════════════════════════════════════════════

FEATURES = ["age", "heart_rate", "LVEF", "DL", "AC", "antiHER2"]

df_cox = df[FEATURES + ["CTRCD", "time"]].dropna()
print(f"\n=== Cox dataset: {len(df_cox)} patients, {df_cox['CTRCD'].sum()} events ===")

# Fit the model
cph = CoxPHFitter(penalizer=0.05)
cph.fit(df_cox, duration_col="time", event_col="CTRCD")

print("\n=== Parsimonious Cox — coefficient table ===")
cph.print_summary(
    decimals=3,
    columns=["coef", "exp(coef)", "se(coef)", "p", "coef lower 95%", "coef upper 95%"],
)

print("\n=== Interpretation ===")
for var, row in cph.summary.iterrows():
    hr = row["exp(coef)"]
    p = row["p"]
    star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
    direction = "increases" if hr > 1 else "decreases"
    print(f"  {var:12s}: HR={hr:.3f} {direction} risk per unit  p={p:.3f} {star}")

# In-sample C-index
c_in = concordance_index(
    df_cox["time"], -cph.predict_partial_hazard(df_cox), df_cox["CTRCD"]
)

# Cross-validated C-index (5-fold, stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, test_idx in skf.split(df_cox, df_cox["CTRCD"]):
    train = df_cox.iloc[train_idx]
    test = df_cox.iloc[test_idx]
    m = CoxPHFitter(penalizer=0.05)
    m.fit(train, duration_col="time", event_col="CTRCD")
    c = concordance_index(test["time"], -m.predict_partial_hazard(test), test["CTRCD"])
    cv_scores.append(c)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"\n=== Performance summary ===")
print(f"  HFA-ICOS AUC (clinical baseline):  {hfa_auc:.3f}")
print(f"  Cox in-sample C-index:             {c_in:.3f}")
print(f"  Cox CV C-index (5-fold):           {cv_mean:.3f} ± {cv_std:.3f}")
print(f"  Improvement over HFA-ICOS:         {(cv_mean - hfa_auc)*100:+.1f} pp")

# ══════════════════════════════════════════════════════════════════════════
# PART 3: Plots
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#fafaf8")

# 3a. Kaplan-Meier curves by HFA-ICOS risk category
ax = axes[0]
cat_colors = {
    "low": "#1D9E75",
    "moderate": "#EF9F27",
    "high": "#D85A30",
    "very_high": "#A32D2D",
}
for cat in cat_order:
    subset = df_scored[df_scored["hfa_category"] == cat]
    if len(subset) == 0:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(subset["time"] / 365, subset["CTRCD"], label=f"{cat} (n={len(subset)})")
    kmf.plot_event_table = False
    ax.plot(
        kmf.timeline,
        1 - kmf.survival_function_.values.flatten(),
        color=cat_colors[cat],
        label=f"{cat} (n={len(subset)})",
        linewidth=1.5,
    )
ax.set_xlabel("Years")
ax.set_ylabel("Cumulative CTRCD incidence")
ax.set_title("HFA-ICOS risk categories", fontweight="500")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# 3b. Cox hazard ratios (forest plot style)
ax = axes[1]
summary = cph.summary.copy()
vars_plot = summary.index.tolist()
y_pos = range(len(vars_plot))
ax.errorbar(
    summary["exp(coef)"],
    y_pos,
    xerr=[
        summary["exp(coef)"] - summary["exp(coef) lower 95%"],
        summary["exp(coef) upper 95%"] - summary["exp(coef)"],
    ],
    fmt="o",
    color="#534AB7",
    ecolor="#AFA9EC",
    capsize=4,
    linewidth=1.5,
)
ax.axvline(1.0, color="#888780", linestyle="--", linewidth=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(vars_plot, fontsize=10)
ax.set_xlabel("Hazard ratio (95% CI)")
ax.set_title("Cox model coefficients", fontweight="500")
ax.spines[["top", "right"]].set_visible(False)

# 3c. Cross-validation performance
ax = axes[2]
models = ["HFA-ICOS\n(clinical\nstandard)", "Cox\n(this model)"]
means = [hfa_auc, cv_mean]
errors = [0, cv_std]
colors = ["#B4B2A9", "#1D9E75"]
bars = ax.bar(models, means, color=colors, alpha=0.85, width=0.5)
ax.errorbar(
    [1],
    [cv_mean],
    yerr=[cv_std],
    fmt="none",
    ecolor="#0F6E56",
    capsize=6,
    linewidth=1.5,
)
ax.set_ylim(0.5, 0.85)
ax.set_ylabel("C-index / AUC")
ax.set_title("Discrimination performance", fontweight="500")
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="500",
    )

plt.tight_layout()
outpath = OUT / "cox_model_results.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved plot → {outpath}")
