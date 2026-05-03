"""
01_eda.py — Exploratory data analysis
Cardio-oncology CTRCD prediction project

Run from the project root:
    python scripts/01_eda.py

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
DATA = Path("data/BC_cardiotox_clinical_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, sep=";", decimal=",")
print(f"Loaded {df.shape[0]} patients, {df.shape[1]} columns")

# ── basic summary ──────────────────────────────────────────────────────────
print("\n=== OUTCOME ===")
print(df["CTRCD"].value_counts())
print(f"CTRCD rate: {df['CTRCD'].mean()*100:.1f}%")

print("\n=== FOLLOW-UP TIME (days) ===")
print(df["time"].describe().round(1))
print(f"Max follow-up: {df['time'].max()/365:.1f} years")

print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n=== CARDIAC VARIABLES ===")
cardiac_cols = ["LVEF", "PWT", "LAd", "LVDd", "LVSd", "heart_rate"]
print(df[cardiac_cols].describe().round(2))

print("\n=== LVEF by outcome ===")
print(df.groupby("CTRCD")["LVEF"].describe().round(2))

print("\n=== AGE by outcome ===")
print(df.groupby("CTRCD")["age"].describe().round(1))

print("\n=== TREATMENT EXPOSURE ===")
treat_cols = ["AC", "antiHER2", "ACprev", "antiHER2prev", "RTprev"]
for col in treat_cols:
    counts = df.groupby("CTRCD")[col].mean() * 100
    print(
        f"  {col}: overall {df[col].mean()*100:.1f}% | "
        f"no-CTRCD {counts[0]:.1f}% | CTRCD {counts[1]:.1f}%"
    )

print("\n=== RISK FACTORS ===")
risk_cols = ["HTA", "DL", "DM", "smoker", "ARRprev"]
for col in risk_cols:
    counts = df.groupby("CTRCD")[col].mean() * 100
    print(
        f"  {col}: overall {df[col].mean()*100:.1f}% | "
        f"no-CTRCD {counts[0]:.1f}% | CTRCD {counts[1]:.1f}%"
    )

# ── plots ──────────────────────────────────────────────────────────────────
c0 = df[df["CTRCD"] == 0]
c1 = df[df["CTRCD"] == 1]
w = 0.35

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#fafaf8")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. LVEF
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(
    c0["LVEF"],
    bins=20,
    alpha=0.65,
    color="#1D9E75",
    label=f"No CTRCD (n={len(c0)})",
    density=True,
)
ax1.hist(
    c1["LVEF"],
    bins=15,
    alpha=0.65,
    color="#D85A30",
    label=f"CTRCD (n={len(c1)})",
    density=True,
)
ax1.axvline(c0["LVEF"].mean(), color="#0F6E56", linestyle="--", linewidth=1.2)
ax1.axvline(c1["LVEF"].mean(), color="#993C1D", linestyle="--", linewidth=1.2)
ax1.set_xlabel("Baseline LVEF (%)")
ax1.set_ylabel("Density")
ax1.set_title("Ejection fraction", fontweight="500")
ax1.legend(fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)

# 2. Age
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(c0["age"], bins=20, alpha=0.65, color="#1D9E75", density=True)
ax2.hist(c1["age"], bins=15, alpha=0.65, color="#D85A30", density=True)
ax2.axvline(c0["age"].mean(), color="#0F6E56", linestyle="--", linewidth=1.2)
ax2.axvline(c1["age"].mean(), color="#993C1D", linestyle="--", linewidth=1.2)
ax2.set_xlabel("Age (years)")
ax2.set_title("Age at diagnosis", fontweight="500")
ax2.spines[["top", "right"]].set_visible(False)

# 3. Follow-up time
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(
    c0["time"] / 365,
    bins=20,
    alpha=0.65,
    color="#1D9E75",
    density=True,
    label="Censored",
)
ax3.hist(
    c1["time"] / 365, bins=15, alpha=0.65, color="#D85A30", density=True, label="Event"
)
ax3.set_xlabel("Time (years)")
ax3.set_title("Follow-up / event time", fontweight="500")
ax3.legend(fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)

# 4. Treatment
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(treat_cols))
treat_labels = [
    "Anthracyclines",
    "Anti-HER2",
    "Prev. anthr.",
    "Prev. antiHER2",
    "Prev. RT",
]
ax4.bar(
    x - w / 2,
    [c0[c].mean() * 100 for c in treat_cols],
    w,
    color="#1D9E75",
    alpha=0.8,
    label="No CTRCD",
)
ax4.bar(
    x + w / 2,
    [c1[c].mean() * 100 for c in treat_cols],
    w,
    color="#D85A30",
    alpha=0.8,
    label="CTRCD",
)
ax4.set_xticks(x)
ax4.set_xticklabels(treat_labels, fontsize=8, rotation=20)
ax4.set_ylabel("% of patients")
ax4.set_title("Treatment exposure", fontweight="500")
ax4.legend(fontsize=9)
ax4.spines[["top", "right"]].set_visible(False)

# 5. Risk factors
ax5 = fig.add_subplot(gs[1, 1])
risk_labels = ["Hypertension", "Dyslipidemia", "Diabetes", "Smoker", "Prev. arrh."]
x = np.arange(len(risk_cols))
ax5.bar(
    x - w / 2, [c0[c].mean() * 100 for c in risk_cols], w, color="#1D9E75", alpha=0.8
)
ax5.bar(
    x + w / 2, [c1[c].mean() * 100 for c in risk_cols], w, color="#D85A30", alpha=0.8
)
ax5.set_xticks(x)
ax5.set_xticklabels(risk_labels, fontsize=8, rotation=20)
ax5.set_ylabel("% of patients")
ax5.set_title("Cardiovascular risk factors", fontweight="500")
ax5.spines[["top", "right"]].set_visible(False)

# 6. Missing data
ax6 = fig.add_subplot(gs[1, 2])
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=True)
bars = ax6.barh(miss.index, miss.values, color="#7F77DD", alpha=0.8)
ax6.set_xlabel("Missing count (of 531)")
ax6.set_title("Missing data by column", fontweight="500")
ax6.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, miss.values):
    ax6.text(
        val + 0.3, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=8
    )

fig.suptitle(
    "BC_cardiotox — exploratory overview", fontsize=14, fontweight="500", y=1.01
)

outpath = OUT / "eda_overview.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved plot → {outpath}")
