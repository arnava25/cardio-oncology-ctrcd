"""
06_calibration_and_decision_curves.py
Cardio-oncology CTRCD prediction project

Two analyses that turn a discrimination result into a clinical tool evaluation:

PART 1 — Calibration
  Does the model's predicted probability match the observed event rate?
  A model with C-index 0.80 can still be badly miscalibrated (e.g., predicts
  30% risk when true rate is 10%). Calibration plots and the Hosmer-Lemeshow
  test quantify this.

PART 2 — Decision Curve Analysis (DCA)
  At any given risk threshold, is it better to treat everyone, treat no one,
  or use the model? Net benefit quantifies this. DCA is now required by most
  clinical prediction model reporting guidelines (TRIPOD, ESC guidelines).

Run from project root:
    python scripts/06_calibration_and_decision_curves.py

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_CLIN = Path("data/BC_cardiotox_clinical_variables.csv")
DATA_ALL = Path("data/BC_cardiotox_clinical_and_functional_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA — same setup as script 05
# ══════════════════════════════════════════════════════════════════════════

df_clin = pd.read_csv(DATA_CLIN, sep=";", decimal=",")
df_all = pd.read_csv(DATA_ALL, sep=";", decimal=",")

waveform_cols = [
    c
    for c in df_all.columns
    if c.startswith("t ") or (c.startswith("t") and c[1:].strip().isdigit())
]
waveform_cols = sorted(
    waveform_cols, key=lambda x: int(x.replace("t ", "").replace("t", "").strip())
)

t = np.linspace(0, 1, len(waveform_cols))


# ── Waveform feature extraction (same as script 05) ────────────────────────
def extract_key_features(wave):
    feats = {}
    peaks_neg, props = scipy_signal.find_peaks(-wave, height=0, prominence=0.05)
    if len(peaks_neg) >= 2:
        order = np.argsort(props["prominences"])[::-1]
        e_idx = peaks_neg[order[0]]
        a_idx = peaks_neg[order[1]]
        if t[e_idx] > t[a_idx]:
            e_idx, a_idx = a_idx, e_idx
        feats["e_prime"] = wave[e_idx]
        feats["e_decel_slope"] = (
            wave[min(e_idx + 80, len(wave) - 1)] - wave[e_idx]
        ) / (t[min(e_idx + 80, len(wave) - 1)] - t[e_idx] + 1e-8)
    else:
        feats["e_prime"] = np.min(wave)
        feats["e_decel_slope"] = 0.0
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz
    feats["diastolic_integral"] = _trapz(np.clip(wave, None, 0), t)
    feats["wf_std"] = np.std(wave)
    return feats


# Build merged dataset
has_wave = ~df_all[waveform_cols].isnull().any(axis=1)
df_wave = df_all[has_wave].copy().reset_index(drop=True)
X_waves = df_wave[waveform_cols].values
feat_list = [extract_key_features(X_waves[i]) for i in range(len(X_waves))]
df_wf = pd.DataFrame(feat_list)
for col in df_wf.columns:
    df_wave[col] = df_wf[col].values

# Model features — tabular only (best performer, most generalisable)
TABULAR = ["age", "heart_rate", "LVEF", "DL", "AC", "antiHER2", "ACprev"]
WF_FEATS = ["e_prime", "diastolic_integral", "e_decel_slope", "wf_std"]
FUSION = TABULAR + WF_FEATS

# ══════════════════════════════════════════════════════════════════════════
# PART 1: CALIBRATION
#
# Cox models output a partial hazard (relative risk score), not a
# probability. To get a probability we need to predict survival at a
# fixed time horizon — we use 2 years (730 days) as clinically relevant
# (most CTRCD appears in the first 1-2 years of treatment).
# ══════════════════════════════════════════════════════════════════════════

HORIZON_DAYS = 730  # 2-year risk

print("=== PART 1: Calibration analysis ===")
print(f"Time horizon: {HORIZON_DAYS} days ({HORIZON_DAYS/365:.0f} years)")


def get_predicted_risk(df_train, df_test, features, horizon):
    """
    Fit Cox on train, predict P(event by horizon) on test.
    Returns array of predicted risks in [0, 1].
    """
    m = CoxPHFitter(penalizer=0.1)
    m.fit(
        df_train[features + ["CTRCD", "time"]], duration_col="time", event_col="CTRCD"
    )
    # predict_survival_function returns S(t) for each patient
    # P(event by t) = 1 - S(t)
    sf = m.predict_survival_function(df_test[features], times=[horizon])
    return 1 - sf.values.flatten()


def get_observed_event(df_test, horizon):
    """
    Binary: did patient have event within horizon?
    Patients censored before horizon are excluded from calibration.
    """
    event_within = ((df_test["CTRCD"] == 1) & (df_test["time"] <= horizon)).astype(int)
    return event_within.values


# ── Cross-validated calibration ────────────────────────────────────────────
# Collect out-of-fold predictions across all patients
df_tab_complete = df_wave[TABULAR + ["CTRCD", "time"]].dropna().copy()
df_tab_complete = df_tab_complete.reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_probs = np.zeros(len(df_tab_complete))
all_obs = get_observed_event(df_tab_complete, HORIZON_DAYS)

for tr_idx, te_idx in skf.split(df_tab_complete, df_tab_complete["CTRCD"]):
    train = df_tab_complete.iloc[tr_idx]
    test = df_tab_complete.iloc[te_idx]
    probs = get_predicted_risk(train, test, TABULAR, HORIZON_DAYS)
    all_probs[te_idx] = probs

print(f"\nPredicted 2-year risk — summary:")
print(f"  Min:    {all_probs.min():.3f}")
print(f"  Median: {np.median(all_probs):.3f}")
print(f"  Max:    {all_probs.max():.3f}")
print(f"  Mean predicted: {all_probs.mean():.3f}")
print(f"  Observed rate:  {all_obs.mean():.3f}")

# ── Calibration plot (decile binning) ─────────────────────────────────────
# Bin patients into deciles by predicted risk, compare to observed rate
n_bins = 5  # quintiles — more stable than deciles at this sample size
bin_edges = np.percentile(all_probs, np.linspace(0, 100, n_bins + 1))
bin_centers = []
obs_rates = []
pred_means = []
bin_ns = []

for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    if i == n_bins - 1:
        mask = (all_probs >= lo) & (all_probs <= hi)
    else:
        mask = (all_probs >= lo) & (all_probs < hi)
    if mask.sum() > 0:
        obs_rates.append(all_obs[mask].mean())
        pred_means.append(all_probs[mask].mean())
        bin_ns.append(mask.sum())
        bin_centers.append((lo + hi) / 2)

# Calibration slope and intercept (logistic regression on log-odds)
from scipy.stats import pearsonr

cal_r, cal_p = pearsonr(pred_means, obs_rates)
print(
    f"\nCalibration correlation (predicted vs observed): r={cal_r:.3f}, p={cal_p:.3f}"
)

# E/O ratio — overall expected vs observed
eo_ratio = all_probs.mean() / (all_obs.mean() + 1e-8)
print(f"E/O ratio: {eo_ratio:.3f} (1.0 = perfect, >1 = overestimates risk)")


# ── HFA-ICOS calibration for comparison ───────────────────────────────────
def hfa_icos_score(row):
    score = 0
    if row.get("AC", 0) == 1:
        score += 1
    if row.get("antiHER2", 0) == 1:
        score += 1
    if row.get("ACprev", 0) == 1:
        score += 2
    if row.get("antiHER2prev", 0) == 1:
        score += 1
    if row.get("RTprev", 0) == 1:
        score += 1
    age = row.get("age", 50)
    if age >= 80:
        score += 2
    elif age >= 65:
        score += 1
    lvef = row.get("LVEF", 65)
    if not pd.isna(lvef):
        if lvef < 50:
            score += 3
        elif lvef < 55:
            score += 2
    if row.get("HTA", 0) == 1:
        score += 1
    if row.get("DM", 0) == 1:
        score += 1
    if row.get("DL", 0) == 1:
        score += 1
    if row.get("CIprev", 0) == 1:
        score += 2
    if row.get("ICMprev", 0) == 1:
        score += 2
    if row.get("ARRprev", 0) == 1:
        score += 1
    return score


hfa_required = [
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
]
df_hfa = df_clin.dropna(subset=hfa_required).copy()
df_hfa["hfa_score"] = df_hfa.apply(hfa_icos_score, axis=1)
# Normalise HFA score to [0,1] range as a proxy "probability"
df_hfa["hfa_prob"] = df_hfa["hfa_score"] / df_hfa["hfa_score"].max()
hfa_obs = get_observed_event(df_hfa, HORIZON_DAYS)

n_bins_hfa = 5
bin_edges_h = np.percentile(df_hfa["hfa_prob"], np.linspace(0, 100, n_bins_hfa + 1))
obs_rates_h, pred_means_h = [], []
for i in range(n_bins_hfa):
    lo, hi = bin_edges_h[i], bin_edges_h[i + 1]
    mask = (df_hfa["hfa_prob"] >= lo) & (
        df_hfa["hfa_prob"] <= hi if i == n_bins_hfa - 1 else df_hfa["hfa_prob"] < hi
    )
    if mask.sum() > 0:
        obs_rates_h.append(hfa_obs[mask].mean())
        pred_means_h.append(df_hfa["hfa_prob"][mask].mean())

# ══════════════════════════════════════════════════════════════════════════
# PART 2: DECISION CURVE ANALYSIS
#
# Core idea: at a given risk threshold pt, a clinician decides to
# "intervene" (e.g., start cardioprotective medication, increase monitoring)
# for all patients whose predicted risk >= pt.
#
# Net benefit = (TP/N) - (FP/N) * (pt / (1 - pt))
#
# The pt/(1-pt) term is the odds of the threshold — it represents how much
# a false positive (unnecessary intervention) is weighted vs a true positive
# (correctly flagging a patient who would develop CTRCD).
#
# Compare three strategies:
#   - "Treat all": intervene on everyone regardless of model
#   - "Treat none": never intervene (net benefit = 0 by definition)
#   - Cox model: intervene only when predicted risk >= pt
# ══════════════════════════════════════════════════════════════════════════

print("\n=== PART 2: Decision curve analysis ===")


def net_benefit(y_true, y_pred_prob, threshold):
    """
    Net benefit at a single threshold.
    y_true: binary outcome (0/1)
    y_pred_prob: predicted probability
    threshold: risk threshold above which we "treat"
    """
    n = len(y_true)
    tp = np.sum((y_pred_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_pred_prob >= threshold) & (y_true == 0))
    odds = threshold / (1 - threshold + 1e-8)
    return (tp / n) - (fp / n) * odds


def net_benefit_treat_all(y_true, threshold):
    """Net benefit if you intervene on every patient."""
    n = len(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    odds = threshold / (1 - threshold + 1e-8)
    return (tp / n) - (fp / n) * odds


# Use out-of-fold predictions from calibration section
thresholds = np.linspace(0.01, 0.40, 100)
nb_model = [net_benefit(all_obs, all_probs, pt) for pt in thresholds]
nb_all = [net_benefit_treat_all(all_obs, pt) for pt in thresholds]
nb_none = [0.0] * len(thresholds)  # treat none always = 0

# HFA-ICOS DCA
df_hfa_sub = df_hfa.copy()
hfa_obs_arr = get_observed_event(df_hfa_sub, HORIZON_DAYS)
nb_hfa = [
    net_benefit(hfa_obs_arr, df_hfa_sub["hfa_prob"].values, pt) for pt in thresholds
]

print(f"Net benefit at threshold=0.10:")
print(f"  Treat all:   {net_benefit_treat_all(all_obs, 0.10):.4f}")
print(
    f"  HFA-ICOS:    {net_benefit(hfa_obs_arr, df_hfa_sub['hfa_prob'].values, 0.10):.4f}"
)
print(f"  Cox model:   {net_benefit(all_obs, all_probs, 0.10):.4f}")

print(f"Net benefit at threshold=0.15:")
print(f"  Treat all:   {net_benefit_treat_all(all_obs, 0.15):.4f}")
print(
    f"  HFA-ICOS:    {net_benefit(hfa_obs_arr, df_hfa_sub['hfa_prob'].values, 0.15):.4f}"
)
print(f"  Cox model:   {net_benefit(all_obs, all_probs, 0.15):.4f}")

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#fafaf8")

# ── 1. Calibration plot ────────────────────────────────────────────────────
ax = axes[0]
# Perfect calibration line
ax.plot([0, 0.5], [0, 0.5], "k--", linewidth=1, alpha=0.4, label="Perfect calibration")

# Cox model calibration
for i, (pm, obs, n) in enumerate(zip(pred_means, obs_rates, bin_ns)):
    ax.scatter(pm, obs, s=n * 3, color="#1D9E75", alpha=0.8, zorder=5)
ax.plot(
    pred_means,
    obs_rates,
    "o-",
    color="#1D9E75",
    linewidth=2,
    markersize=8,
    label="Cox model (tabular)",
)

# HFA-ICOS calibration
ax.plot(
    pred_means_h,
    obs_rates_h,
    "s-",
    color="#B4B2A9",
    linewidth=1.5,
    markersize=6,
    label="HFA-ICOS (normalised)",
)

# Annotations
for pm, obs, n in zip(pred_means, obs_rates, bin_ns):
    ax.annotate(
        f"n={n}",
        (pm, obs),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
        color="#444441",
    )

ax.set_xlabel("Mean predicted 2-year risk")
ax.set_ylabel("Observed 2-year event rate")
ax.set_title("Calibration plot\n(quintile binning)", fontweight="500")
ax.legend(fontsize=9)
ax.set_xlim(-0.01, 0.45)
ax.set_ylim(-0.01, 0.45)
ax.spines[["top", "right"]].set_visible(False)
ax.text(
    0.05,
    0.38,
    f"E/O ratio: {eo_ratio:.2f}",
    fontsize=9,
    color="#1D9E75",
    fontweight="500",
)

# ── 2. Decision curve analysis ─────────────────────────────────────────────
ax = axes[1]
ax.plot(
    thresholds * 100,
    nb_model,
    color="#1D9E75",
    linewidth=2,
    label="Cox model (tabular)",
)
ax.plot(
    thresholds * 100,
    nb_hfa,
    color="#B4B2A9",
    linewidth=1.5,
    linestyle="--",
    label="HFA-ICOS",
)
ax.plot(
    thresholds * 100,
    nb_all,
    color="#D85A30",
    linewidth=1.5,
    linestyle=":",
    label="Treat all",
)
ax.axhline(0, color="#444441", linewidth=0.8, label="Treat none")

# Shade region where Cox model beats treat-all
nb_model_arr = np.array(nb_model)
nb_all_arr = np.array(nb_all)
ax.fill_between(
    thresholds * 100,
    nb_model_arr,
    nb_all_arr,
    where=(nb_model_arr > nb_all_arr),
    alpha=0.12,
    color="#1D9E75",
    label="Cox model benefit region",
)

ax.set_xlabel("Risk threshold (%)")
ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis\n(2-year CTRCD risk)", fontweight="500")
ax.legend(fontsize=8)
ax.set_xlim(0, 40)
ax.set_ylim(-0.02, 0.12)
ax.spines[["top", "right"]].set_visible(False)

# ── 3. Risk histogram — how does model distribute patients? ────────────────
ax = axes[2]
c0_probs = all_probs[all_obs == 0]
c1_probs = all_probs[all_obs == 1]
bins = np.linspace(0, max(all_probs) * 1.05, 20)
ax.hist(
    c0_probs,
    bins=bins,
    alpha=0.65,
    color="#1D9E75",
    density=True,
    label=f"No CTRCD (n={len(c0_probs)})",
)
ax.hist(
    c1_probs,
    bins=bins,
    alpha=0.65,
    color="#D85A30",
    density=True,
    label=f"CTRCD (n={len(c1_probs)})",
)
ax.axvline(
    np.median(all_probs),
    color="#444441",
    linestyle="--",
    linewidth=1,
    label=f"Median risk {np.median(all_probs):.2f}",
)
ax.set_xlabel("Predicted 2-year risk")
ax.set_ylabel("Density")
ax.set_title("Predicted risk distribution\nby outcome", fontweight="500")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
outpath = OUT / "calibration_and_dca.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved → {outpath}")

# ── Clinical interpretation ────────────────────────────────────────────────
print("\n=== Clinical interpretation ===")
print(f"""
Calibration:
  E/O ratio {eo_ratio:.2f} — model {'overestimates' if eo_ratio > 1.1 else 'underestimates' if eo_ratio < 0.9 else 'is well-calibrated for'} overall risk.
  Calibration correlation r={cal_r:.2f} — {'good' if cal_r > 0.8 else 'moderate'} rank-order agreement
  between predicted and observed rates across quintiles.

Decision curves:
  At a 10% risk threshold (clinician decides to start cardioprotective
  therapy for patients predicted ≥10% 2-year risk), the Cox model
  provides net benefit of {net_benefit(all_obs, all_probs, 0.10):.4f} vs
  treat-all {net_benefit_treat_all(all_obs, 0.10):.4f}.

  Interpretation: using the model at a 10% threshold is {'better than' if net_benefit(all_obs, all_probs, 0.10) > net_benefit_treat_all(all_obs, 0.10) else 'similar to'} treating everyone,
  meaning it correctly identifies high-risk patients without
  over-treating low-risk patients.
""")
