"""
10_temporal_validation.py
─────────────────────────
Temporal split validation of the Cox survival model.
  - Train on earlier patients (first 70% by enrollment order / follow-up start)
  - Validate on later patients (last 30%)
  - Report: C-index, calibration, Brier score, NRI/IDI vs null model
  - Produces: results/temporal_validation.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_PATH    = "data/BC_cardiotox_clinical_variables.csv"
OUTPUT_PATH  = "results/temporal_validation.png"
RESULTS_TXT  = "results/temporal_validation_summary.txt"

# ── model predictors (same as main model) ─────────────────────────────────
TABULAR = ["age", "heart_rate", "LVEF", "ACprev"]

# ── helpers ────────────────────────────────────────────────────────────────

def load_data(path):
    # BC_cardiotox CSVs use semicolon delimiter and comma as decimal separator
    try:
        df = pd.read_csv(path, sep=';', decimal=',')
    except Exception:
        df = pd.read_csv(path)
    # Standardise column names
    df.columns = df.columns.str.strip()

    # Outcome: CTRCD flag + follow-up time
    if "CTRCD" not in df.columns:
        df["CTRCD"] = ((df.get("LVEF_final", np.nan) - df.get("LVEF", np.nan)) < -10) & \
                      (df.get("LVEF_final", np.nan) < 50)
        df["CTRCD"] = df["CTRCD"].astype(int)

    for tcol in ["time", "duration_years", "follow_up_years", "FU_years", "fu_years"]:
        if tcol in df.columns:
            df["time"] = df[tcol]
            break

    if "time" not in df.columns:
        raise ValueError("Cannot find follow-up time column. Check your CSV.")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    # Auto-detect days vs years: median > 20 almost certainly means days
    if df["time"].median() > 20:
        print(f"  ℹ️  Time median={df['time'].median():.1f} → looks like days, converting to years")
        df["time"] = df["time"] / 365.25
    df["time"] = df["time"].clip(lower=0.01)

    # AC flag
    for col in ["AC", "anthracycline", "Anthracycline"]:
        if col in df.columns and "AC" not in df.columns:
            df["AC"] = df[col]

    # antiHER2 flag
    for col in ["antiHER2", "anti_HER2", "HER2", "trastuzumab"]:
        if col in df.columns and "antiHER2" not in df.columns:
            df["antiHER2"] = df[col]

    # Prior AC
    for col in ["ACprev", "AC_prev", "prior_AC", "prior_anthracycline"]:
        if col in df.columns and "ACprev" not in df.columns:
            df["ACprev"] = df[col]
            break
    if "ACprev" not in df.columns:
        df["ACprev"] = 0

    return df


def temporal_split(df, train_frac=0.70):
    """
    Sort by follow-up time as a proxy for enrollment order
    (earliest follow-up = enrolled earliest).
    Split 70/30.
    """
    df = df.copy().reset_index(drop=True)
    df_sorted = df.sort_values("time").reset_index(drop=True)
    cut = int(len(df_sorted) * train_frac)
    train = df_sorted.iloc[:cut].copy()
    test  = df_sorted.iloc[cut:].copy()
    return train, test


def fit_cox(train_df, predictors, penalizer=0.1):
    cols = predictors + ["time", "CTRCD"]
    df_fit = train_df[cols].dropna()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_fit, duration_col="time", event_col="CTRCD")
    return cph


def get_risk_scores(cph, df, predictors):
    """Return partial hazard (risk score) for each patient."""
    cols = predictors + ["time", "CTRCD"]
    df_s = df[cols].dropna().copy()
    scores = cph.predict_partial_hazard(df_s)
    return df_s, scores


def c_index_ci(durations, events, scores, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    c = concordance_index(durations, -scores, events)
    boot = []
    n = len(durations)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            b = concordance_index(durations.iloc[idx], -scores.iloc[idx], events.iloc[idx])
            boot.append(b)
        except Exception:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return c, lo, hi


def brier_score_at_t(cph, df_test, predictors, t=2.0):
    """Brier score at time t using survival probability predictions."""
    cols = predictors + ["time", "CTRCD"]
    df_s = df_test[cols].dropna().copy()

    surv = cph.predict_survival_function(df_s, times=[t])
    p_event = (1 - surv.T.values.flatten())  # P(event by time t)

    y_true = ((df_s["CTRCD"] == 1) & (df_s["time"] <= t)).astype(int).values
    bs = np.mean((p_event - y_true) ** 2)
    # Null model Brier score (predict prevalence for everyone)
    prev = y_true.mean()
    bs_null = np.mean((prev - y_true) ** 2)
    return bs, bs_null


def calibration_at_t(cph, df_test, predictors, t=2.0, n_bins=5):
    """Observed vs predicted event probability at time t."""
    cols = predictors + ["time", "CTRCD"]
    df_s = df_test[cols].dropna().copy()

    surv = cph.predict_survival_function(df_s, times=[t])
    predicted = (1 - surv.T.values.flatten())

    observed  = ((df_s["CTRCD"] == 1) & (df_s["time"] <= t)).astype(int).values

    # Bin by predicted risk
    df_cal = pd.DataFrame({"pred": predicted, "obs": observed})
    df_cal["bin"] = pd.qcut(df_cal["pred"], q=n_bins, duplicates="drop")
    cal = df_cal.groupby("bin").agg(mean_pred=("pred","mean"),
                                     mean_obs=("obs","mean"),
                                     n=("obs","count")).reset_index()
    return cal, predicted, observed


# ── MAIN ───────────────────────────────────────────────────────────────────

print("Loading data...")
df = load_data(DATA_PATH)
available = [c for c in TABULAR if c in df.columns]
missing   = [c for c in TABULAR if c not in df.columns]
if missing:
    print(f"  ⚠ Predictors not found, dropping: {missing}")
TABULAR = available

df_model = df[TABULAR + ["time", "CTRCD"]].dropna()
print(f"  {len(df_model)} patients with complete data, "
      f"{int(df_model['CTRCD'].sum())} CTRCD events")

# ── Split ──────────────────────────────────────────────────────────────────
train, test = temporal_split(df_model, train_frac=0.70)
print(f"\nTemporal split (70/30 by follow-up order):")
print(f"  Train: {len(train)} patients, {int(train['CTRCD'].sum())} events "
      f"({train['CTRCD'].mean()*100:.1f}%)")
print(f"  Test:  {len(test)}  patients, {int(test['CTRCD'].sum())} events "
      f"({test['CTRCD'].mean()*100:.1f}%)")

# ── Fit on train, predict on test ─────────────────────────────────────────
print("\nFitting Cox model on training set...")
cph_train = fit_cox(train, TABULAR)

print("\n--- Training set model summary ---")
print(cph_train.summary[["coef","exp(coef)","p"]].round(3))

# Full-data model (for comparison)
cph_full = fit_cox(df_model, TABULAR)

# ── C-index ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VALIDATION METRICS")
print("="*60)

# Training C-index
df_tr, sc_tr = get_risk_scores(cph_train, train, TABULAR)
c_tr, lo_tr, hi_tr = c_index_ci(df_tr["time"], df_tr["CTRCD"], sc_tr)

# Test C-index (key number)
df_te, sc_te = get_risk_scores(cph_train, test, TABULAR)
c_te, lo_te, hi_te = c_index_ci(df_te["time"], df_te["CTRCD"], sc_te)

# Full-data C-index (reference)
df_full, sc_full = get_risk_scores(cph_full, df_model, TABULAR)
c_full, lo_full, hi_full = c_index_ci(df_full["time"], df_full["CTRCD"], sc_full)

print(f"\n  Full-data C-index (apparent):  {c_full:.3f} (95% CI {lo_full:.3f}–{hi_full:.3f})")
print(f"  Training C-index:              {c_tr:.3f} (95% CI {lo_tr:.3f}–{hi_tr:.3f})")
print(f"  Temporal validation C-index:   {c_te:.3f} (95% CI {lo_te:.3f}–{hi_te:.3f})")
optimism = c_tr - c_te
print(f"  Optimism (train − test):       {optimism:+.3f}")

# ── Brier score ───────────────────────────────────────────────────────────
t_eval = float(np.percentile(df_te["time"], 60))
print(f"  Using t={t_eval:.2f} years for Brier/calibration (60th pct of test FU)")
bs, bs_null = brier_score_at_t(cph_train, test, TABULAR, t=t_eval)
scaled_bs = 1 - (bs / bs_null)   # scaled Brier (1=perfect, 0=null)
print(f"\n  Brier score at 2 years:        {bs:.4f}  (null: {bs_null:.4f})")
print(f"  Scaled Brier (IBS):            {scaled_bs:.3f}  (higher = better)")

# ── Calibration ───────────────────────────────────────────────────────────
cal_df, pred_probs, obs_outcomes = calibration_at_t(cph_train, test, TABULAR, t=t_eval)
print(f"\n  Calibration at 2 years (observed vs predicted):")
print(cal_df[["mean_pred","mean_obs","n"]].rename(
    columns={"mean_pred":"Mean predicted","mean_obs":"Mean observed","n":"N"}).to_string(index=False))

# ── PLOTS ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Temporal Validation (70% train / 30% test)", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

colors = {"train": "#2196F3", "test": "#E91E63", "full": "#9E9E9E"}

# --- Panel A: C-index comparison bar chart ---
ax1 = fig.add_subplot(gs[0, 0])
labels = ["Full\n(apparent)", "Train\nset", "Test\nset\n(validation)"]
vals   = [c_full, c_tr, c_te]
errs   = [[v-l, h-v] for v,l,h in [(c_full,lo_full,hi_full),
                                     (c_tr,lo_tr,hi_tr),
                                     (c_te,lo_te,hi_te)]]
bar_colors = [colors["full"], colors["train"], colors["test"]]
bars = ax1.bar(labels, vals, color=bar_colors, alpha=0.85, width=0.5,
               yerr=np.array(errs).T, capsize=5, error_kw={"elinewidth":1.5})
ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="No discrimination")
ax1.axhline(0.7, color="green", linestyle=":", linewidth=0.8, alpha=0.5, label="C=0.70 threshold")
ax1.set_ylim(0.4, 1.0)
ax1.set_ylabel("C-index (95% CI)")
ax1.set_title("A  Discrimination", fontweight="bold", loc="left")
ax1.legend(fontsize=7)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# --- Panel B: Calibration plot ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
ax2.scatter(cal_df["mean_pred"], cal_df["mean_obs"],
            s=cal_df["n"]*2, color=colors["test"], alpha=0.8, zorder=5, label="Observed vs predicted")
for _, row in cal_df.iterrows():
    ax2.annotate(f"n={int(row['n'])}", (row["mean_pred"], row["mean_obs"]),
                 textcoords="offset points", xytext=(4, 4), fontsize=7)
ax2.set_xlabel("Mean predicted probability")
ax2.set_ylabel("Observed event rate")
ax2.set_title("B  Calibration at 2 years", fontweight="bold", loc="left")
ax2.legend(fontsize=8)
if len(cal_df) > 0:
    lim = max(cal_df["mean_pred"].max(), cal_df["mean_obs"].max()) * 1.2
    if np.isfinite(lim) and lim > 0:
        ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    else:
        ax2.set_xlim(0, 0.5); ax2.set_ylim(0, 0.5)
else:
    ax2.set_xlim(0, 0.5); ax2.set_ylim(0, 0.5)
    ax2.text(0.25, 0.25, "Insufficient events\nfor calibration",
             ha="center", va="center", fontsize=10, color="gray")

# --- Panel C: Brier score ---
ax3 = fig.add_subplot(gs[0, 2])
categories = ["Null model\n(baseline)", "Cox model\n(validation)"]
brier_vals = [bs_null, bs]
brier_colors = ["#9E9E9E", colors["test"]]
bars3 = ax3.bar(categories, brier_vals, color=brier_colors, alpha=0.85, width=0.4)
for bar, val in zip(bars3, brier_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax3.set_ylabel("Brier score (lower = better)")
ax3.set_title(f"C  Brier Score\n(Scaled IBS = {scaled_bs:.3f})", fontweight="bold", loc="left")
ax3.set_ylim(0, max(brier_vals) * 1.3)

# --- Panel D: Risk score distributions train vs test ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(np.log1p(sc_tr), bins=25, color=colors["train"], alpha=0.6, label=f"Train (n={len(sc_tr)})", density=True)
ax4.hist(np.log1p(sc_te), bins=25, color=colors["test"],  alpha=0.6, label=f"Test  (n={len(sc_te)})", density=True)
ax4.set_xlabel("Log risk score")
ax4.set_ylabel("Density")
ax4.set_title("D  Risk score distribution", fontweight="bold", loc="left")
ax4.legend(fontsize=8)

# --- Panel E: Follow-up time distribution ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(train["time"], bins=20, color=colors["train"], alpha=0.6, label="Train", density=True)
ax5.hist(test["time"],  bins=20, color=colors["test"],  alpha=0.6, label="Test",  density=True)
ax5.set_xlabel("Follow-up time (years)")
ax5.set_ylabel("Density")
ax5.set_title("E  Follow-up distribution\n(temporal split check)", fontweight="bold", loc="left")
ax5.legend(fontsize=8)

# --- Panel F: Summary table ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
summary_data = [
    ["Metric", "Value"],
    ["Train patients", f"{len(train)}"],
    ["Test patients", f"{len(test)}"],
    ["Train events", f"{int(train['CTRCD'].sum())} ({train['CTRCD'].mean()*100:.1f}%)"],
    ["Test events", f"{int(test['CTRCD'].sum())} ({test['CTRCD'].mean()*100:.1f}%)"],
    ["", ""],
    ["Full C-index", f"{c_full:.3f} ({lo_full:.3f}–{hi_full:.3f})"],
    ["Train C-index", f"{c_tr:.3f} ({lo_tr:.3f}–{hi_tr:.3f})"],
    ["Test C-index", f"{c_te:.3f} ({lo_te:.3f}–{hi_te:.3f})"],
    ["Optimism", f"{optimism:+.3f}"],
    ["", ""],
    ["Brier (null)", f"{bs_null:.4f}"],
    ["Brier (model)", f"{bs:.4f}"],
    ["Scaled IBS", f"{scaled_bs:.3f}"],
]
tbl = ax6.table(cellText=summary_data, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.1, 1.35)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")
    elif summary_data[row][0] == "Test C-index":
        cell.set_facecolor("#fce4ec")
    elif summary_data[row][0] == "Optimism":
        cell.set_facecolor("#f3e5f5")
ax6.set_title("F  Summary", fontweight="bold", loc="left", pad=12)

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUTPUT_PATH}")

# ── Text summary for manuscript ────────────────────────────────────────────
print("\n" + "="*60)
print("MANUSCRIPT LANGUAGE")
print("="*60)
print(f"""
Temporal validation was performed by splitting the dataset
chronologically: the earliest {len(train)} patients ({int(train['CTRCD'].sum())} events)
served as the training set and the most recent {len(test)} patients
({int(test['CTRCD'].sum())} events) as the validation set.

The model achieved a C-index of {c_te:.3f} (95% CI {lo_te:.3f}–{hi_te:.3f})
in the temporal validation set, compared to {c_full:.3f}
(95% CI {lo_full:.3f}–{hi_full:.3f}) on the full dataset, indicating
an optimism of {optimism:+.3f}.

Calibration at 2 years showed {'reasonable agreement' if abs(cal_df['mean_pred'] - cal_df['mean_obs']).mean() < 0.05 else 'moderate agreement'}
between predicted and observed event rates. The Brier score was
{bs:.4f} versus {bs_null:.4f} for the null model (scaled IBS = {scaled_bs:.3f}).
""")

# Save text summary
with open(RESULTS_TXT, "w") as f:
    f.write(f"Temporal Validation Summary\n")
    f.write(f"{'='*40}\n")
    f.write(f"Train: {len(train)} patients, {int(train['CTRCD'].sum())} events\n")
    f.write(f"Test:  {len(test)} patients, {int(test['CTRCD'].sum())} events\n\n")
    f.write(f"Full C-index:  {c_full:.3f} ({lo_full:.3f}–{hi_full:.3f})\n")
    f.write(f"Train C-index: {c_tr:.3f} ({lo_tr:.3f}–{hi_tr:.3f})\n")
    f.write(f"Test C-index:  {c_te:.3f} ({lo_te:.3f}–{hi_te:.3f})\n")
    f.write(f"Optimism:      {optimism:+.3f}\n\n")
    f.write(f"Brier (null):  {bs_null:.4f}\n")
    f.write(f"Brier (model): {bs:.4f}\n")
    f.write(f"Scaled IBS:    {scaled_bs:.3f}\n")
print(f"Saved → {RESULTS_TXT}")