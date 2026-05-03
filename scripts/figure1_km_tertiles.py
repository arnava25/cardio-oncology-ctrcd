
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data/BC_cardiotox_clinical_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, sep=";", decimal=",")
df["time"] = df["time"] / 365.25

TABULAR = ["age", "heart_rate", "LVEF", "ACprev"]
df2 = df[TABULAR + ["time", "CTRCD"]].dropna().reset_index(drop=True)

# Cross-validated OOF risk scores
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_scores = np.zeros(len(df2))
for train_idx, val_idx in skf.split(df2, df2["CTRCD"]):
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df2.iloc[train_idx], duration_col="time", event_col="CTRCD")
    oof_scores[val_idx] = cph.predict_partial_hazard(df2.iloc[val_idx])

df2["tertile"] = pd.qcut(oof_scores, q=3, labels=["Low", "Intermediate", "High"])

# Print incidence at supported time horizons
print("Cumulative incidence by tertile:")
for t in ["Low", "Intermediate", "High"]:
    grp = df2[df2["tertile"] == t]
    kmf = KaplanMeierFitter()
    kmf.fit(grp["time"], event_observed=grp["CTRCD"])
    for yr in [2, 3, 5]:
        if grp["time"].max() >= yr:
            ci = 1 - kmf.survival_function_at_times([yr]).values[0]
            at_risk = (grp["time"] >= yr).sum()
            print(f"  {t} {yr}yr: {ci*100:.1f}% (at risk={at_risk})")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#fafaf8")

colors = {"Low": "#5B9BD5", "Intermediate": "#F4A030", "High": "#C0392B"}
n_events = df2.groupby("tertile")["CTRCD"].sum()
n_total = df2.groupby("tertile")["CTRCD"].count()

for tertile in ["Low", "Intermediate", "High"]:
    grp = df2[df2["tertile"] == tertile]
    kmf = KaplanMeierFitter()
    kmf.fit(grp["time"], event_observed=grp["CTRCD"])

    # Plot cumulative incidence = 1 - KM survival
    t_vals = kmf.survival_function_.index.values
    ci_vals = 1 - kmf.survival_function_.values.flatten()

    # Confidence bands

    ci_cols = kmf.confidence_interval_.columns
    ci_lower = 1 - kmf.confidence_interval_[ci_cols[1]].values
    ci_upper = 1 - kmf.confidence_interval_[ci_cols[0]].values

    n = n_total[tertile]
    ev = int(n_events[tertile])
    label = f"{tertile} risk (n={n}, ev={ev})"

    ax.plot(t_vals, ci_vals, color=colors[tertile], linewidth=2, label=label)
    ax.fill_between(t_vals, ci_lower, ci_upper, alpha=0.15, color=colors[tertile])

# Truncate to 6 years
ax.set_xlim(0, 6)
ax.set_ylim(0, 0.55)
ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Cumulative CTRCD incidence", fontsize=12)
ax.set_title("CTRCD cumulative incidence by predicted risk tertile", fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
outpath = OUT / "figure1_km_tertiles.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved → {outpath}")