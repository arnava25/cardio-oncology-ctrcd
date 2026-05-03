"""
05_fusion_model.py — Fused clinical + TDI waveform model
Cardio-oncology CTRCD prediction project

Combines the best tabular Cox model (Model B: age, heart_rate, LVEF,
DL, AC, antiHER2, ACprev) with waveform-derived features, evaluated
on the 270-patient subset that has both data types.

Three questions this script answers:
  1. Does adding waveform features to the clinical model improve C-index?
  2. Which waveform features add the most beyond what tabular data already captures?
  3. What does the full patient risk trajectory look like?

Run from project root:
    python scripts/05_fusion_model.py

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import mannwhitneyu
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_CLIN = Path("data/BC_cardiotox_clinical_variables.csv")
DATA_FUNC = Path("data/BC_cardiotox_functional_variable.csv")
DATA_ALL = Path("data/BC_cardiotox_clinical_and_functional_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# PART 1: Load and merge clinical + waveform data
# ══════════════════════════════════════════════════════════════════════════

print("=== Loading data ===")
df_clin = pd.read_csv(DATA_CLIN, sep=";", decimal=",")
df_func = pd.read_csv(DATA_FUNC, sep=";", decimal=",")

# The combined file has both — use it to keep patient alignment correct
df_all = pd.read_csv(DATA_ALL, sep=";", decimal=",")
print(f"Combined dataset: {df_all.shape[0]} patients, {df_all.shape[1]} columns")

# Separate waveform columns from clinical columns
waveform_cols = [c for c in df_all.columns if c.startswith('t ') or
                 (c.startswith('t') and c[1:].strip().isdigit())]
waveform_cols = sorted(waveform_cols,
                       key=lambda x: int(x.replace('t ', '').replace('t','').strip()))

clinical_cols = [c for c in df_all.columns if c not in waveform_cols]
print(
    f"Clinical columns: {len(clinical_cols)}, Waveform timepoints: {len(waveform_cols)}"
)

# Patients with complete waveforms
has_waveform = ~df_all[waveform_cols].isnull().any(axis=1)
df_wave = df_all[has_waveform].copy().reset_index(drop=True)
print(
    f"Patients with complete waveforms: {len(df_wave)} "
    f"(events={df_wave['CTRCD'].sum()})"
)

# ══════════════════════════════════════════════════════════════════════════
# PART 2: Extract waveform features for each patient
# Focus on features with biological meaning and statistical signal from script 04
# ══════════════════════════════════════════════════════════════════════════

t = np.linspace(0, 1, len(waveform_cols))


def extract_key_features(wave):
    """
    Extract the waveform features that showed signal in script 04,
    plus a few additional diastolic markers.
    """
    feats = {}

    # ── e' wave: early diastolic velocity ─────────────────────────────────
    # Most important finding: CTRCD patients have reduced e' magnitude
    # (slower myocardial relaxation = subclinical diastolic dysfunction)
    peaks_neg, props = scipy_signal.find_peaks(-wave, height=0, prominence=0.05)
    if len(peaks_neg) >= 2:
        order = np.argsort(props["prominences"])[::-1]
        e_idx = peaks_neg[order[0]]
        a_idx = peaks_neg[order[1]]
        if t[e_idx] > t[a_idx]:
            e_idx, a_idx = a_idx, e_idx
        feats["e_prime"] = wave[e_idx]  # negative: lower = better relaxation
        feats["a_prime"] = wave[a_idx]
        feats["e_a_ratio"] = abs(wave[e_idx]) / (abs(wave[a_idx]) + 1e-6)
        feats["e_prime_t"] = t[e_idx]  # timing of peak relaxation
    elif len(peaks_neg) == 1:
        feats["e_prime"] = wave[peaks_neg[0]]
        feats["a_prime"] = wave[peaks_neg[0]]
        feats["e_a_ratio"] = 1.0
        feats["e_prime_t"] = t[peaks_neg[0]]
    else:
        feats["e_prime"] = np.min(wave)
        feats["a_prime"] = np.min(wave)
        feats["e_a_ratio"] = 1.0
        feats["e_prime_t"] = t[np.argmin(wave)]

    # ── s' wave: systolic peak velocity ───────────────────────────────────
    peaks_pos, props_pos = scipy_signal.find_peaks(wave, height=0, prominence=0.1)
    if len(peaks_pos) > 0:
        best = peaks_pos[np.argmax(props_pos["prominences"])]
        feats["s_prime"] = wave[best]
        feats["s_prime_t"] = t[best]
    else:
        feats["s_prime"] = np.max(wave)
        feats["s_prime_t"] = t[np.argmax(wave)]

    # ── diastolic integral ────────────────────────────────────────────────
    # Area under the negative portion of the waveform (diastolic phase)
    # Reduced magnitude = impaired diastolic function
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz
    neg_wave = np.clip(wave, None, 0)
    feats["diastolic_integral"] = _trapz(neg_wave, t)  # negative value

    # ── diastolic deceleration slope ──────────────────────────────────────
    # How quickly does the e' wave decelerate after its peak?
    # Steeper = more abrupt relaxation termination
    if len(peaks_neg) >= 1:
        e_idx_local = (
            peaks_neg[np.argmax(props["prominences"])]
            if len(peaks_neg) > 0
            else np.argmin(wave)
        )
        end_idx = min(e_idx_local + 80, len(wave) - 1)
        if end_idx > e_idx_local:
            slope = (wave[end_idx] - wave[e_idx_local]) / (
                t[end_idx] - t[e_idx_local] + 1e-8
            )
            feats["e_decel_slope"] = slope
        else:
            feats["e_decel_slope"] = 0.0
    else:
        feats["e_decel_slope"] = 0.0

    # ── overall waveform energy ratio ─────────────────────────────────────
    fft_mag = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave))
    feats["energy_ratio"] = np.sum(fft_mag[freqs < 0.05] ** 2) / (
        np.sum(fft_mag[freqs >= 0.05] ** 2) + 1e-6
    )

    # ── waveform standard deviation ───────────────────────────────────────
    feats["wf_std"] = np.std(wave)

    return feats


print("\n=== Extracting waveform features ===")
X_waves = df_wave[waveform_cols].values
feat_list = [extract_key_features(X_waves[i]) for i in range(len(X_waves))]
df_wf = pd.DataFrame(feat_list)
print(f"Extracted {len(df_wf.columns)} features: {list(df_wf.columns)}")

# ── Check which waveform features differ by outcome ────────────────────────
print("\n=== Waveform features vs outcome (on 270-patient subset) ===")
wf_feature_names = list(df_wf.columns)
for feat in wf_feature_names:
    g0 = df_wf.loc[df_wave["CTRCD"].values == 0, feat]
    g1 = df_wf.loc[df_wave["CTRCD"].values == 1, feat]
    _, p = mannwhitneyu(g0, g1, alternative="two-sided")
    sig = " ***" if p < 0.01 else (" **" if p < 0.05 else (" *" if p < 0.1 else ""))
    print(
        f"  {feat:20s}: no-CTRCD {g0.mean():7.3f} | CTRCD {g1.mean():7.3f} | p={p:.3f}{sig}"
    )

# ══════════════════════════════════════════════════════════════════════════
# PART 3: Build and compare models on the 270-patient waveform subset
# ══════════════════════════════════════════════════════════════════════════

# Attach waveform features to clinical data
df_merged = df_wave[clinical_cols].copy().reset_index(drop=True)
for col in wf_feature_names:
    df_merged[col] = df_wf[col].values

print(
    f"\n=== Merged dataset: {len(df_merged)} patients, "
    f"{df_merged['CTRCD'].sum()} events ==="
)

# Clinical variables (Model B from script 02b — best tabular model)
TABULAR_FEATURES = ["age", "heart_rate", "LVEF", "DL", "AC", "antiHER2", "ACprev"]

# Waveform features to add — use those with p < 0.2 from above + e_prime always
# e_prime is the key finding regardless of p-value in this small subset
WF_FEATURES = ["e_prime", "diastolic_integral", "e_decel_slope", "wf_std"]

FUSION_FEATURES = TABULAR_FEATURES + WF_FEATURES


def run_cv(df_data, features, n_splits=5, penalizer=0.1, seed=42):
    """5-fold stratified CV, returns (mean, std, per-fold scores)."""
    needed = features + ["CTRCD", "time"]
    df_sub = df_data[needed].dropna()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in skf.split(df_sub, df_sub["CTRCD"]):
        train = df_sub.iloc[tr_idx]
        test = df_sub.iloc[te_idx]
        m = CoxPHFitter(penalizer=penalizer)
        m.fit(train, duration_col="time", event_col="CTRCD")
        c = concordance_index(
            test["time"], -m.predict_partial_hazard(test[features]), test["CTRCD"]
        )
        scores.append(c)
    n = len(df_sub)
    events = int(df_sub["CTRCD"].sum())
    return np.mean(scores), np.std(scores), scores, n, events


print("\n=== Cross-validated C-index comparison (all on 270-patient subset) ===")

print("\nHFA-ICOS clinical baseline (reference): 0.663")

print("\nTabular only (Model B):")
m_tab_mean, m_tab_std, m_tab_folds, n_tab, ev_tab = run_cv(df_merged, TABULAR_FEATURES)
print(f"  N={n_tab}, events={ev_tab}")
print(f"  CV C-index: {m_tab_mean:.3f} ± {m_tab_std:.3f}")

print("\nWaveform features only:")
m_wf_mean, m_wf_std, m_wf_folds, n_wf, ev_wf = run_cv(df_merged, WF_FEATURES)
print(f"  N={n_wf}, events={ev_wf}")
print(f"  CV C-index: {m_wf_mean:.3f} ± {m_wf_std:.3f}")

print("\nFusion (tabular + waveform):")
m_fus_mean, m_fus_std, m_fus_folds, n_fus, ev_fus = run_cv(df_merged, FUSION_FEATURES)
print(f"  N={n_fus}, events={ev_fus}")
print(f"  CV C-index: {m_fus_mean:.3f} ± {m_fus_std:.3f}")

print(f"\n=== Net gain from adding waveform to tabular ===")
print(f"  Tabular:  {m_tab_mean:.3f}")
print(f"  Fusion:   {m_fus_mean:.3f}")
print(f"  Delta:    {(m_fus_mean - m_tab_mean)*100:+.1f} pp")

# ── Fit fusion model on full subset for coefficient inspection ─────────────
print("\n=== Fusion model coefficient table ===")
df_fus_complete = df_merged[FUSION_FEATURES + ["CTRCD", "time"]].dropna()
cph_fus = CoxPHFitter(penalizer=0.1)
cph_fus.fit(df_fus_complete, duration_col="time", event_col="CTRCD")
cph_fus.print_summary(decimals=3, columns=["coef", "exp(coef)", "se(coef)", "p"])

# ══════════════════════════════════════════════════════════════════════════
# PART 4: Risk score and patient trajectories
# ══════════════════════════════════════════════════════════════════════════

# Generate predicted risk scores for all patients with complete data
df_fus_complete = df_fus_complete.copy()
df_fus_complete["risk_score"] = cph_fus.predict_partial_hazard(
    df_fus_complete[FUSION_FEATURES]
)

# Tertile risk groups
tertiles = df_fus_complete["risk_score"].quantile([1 / 3, 2 / 3])


def assign_tertile(score):
    if score <= tertiles.iloc[0]:
        return "low"
    elif score <= tertiles.iloc[1]:
        return "medium"
    else:
        return "high"


df_fus_complete["risk_group"] = df_fus_complete["risk_score"].apply(assign_tertile)

print("\n=== CTRCD rate by fusion model risk tertile ===")
tert_summary = df_fus_complete.groupby("risk_group").agg(
    n=("CTRCD", "count"),
    events=("CTRCD", "sum"),
    rate=("CTRCD", lambda x: f"{x.mean()*100:.1f}%"),
)
print(tert_summary.loc[["low", "medium", "high"]])

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#fafaf8")

# 1. Per-fold C-index comparison
ax = axes[0]
x = np.arange(1, 6)
ax.plot(
    x,
    m_tab_folds,
    "o-",
    color="#9FE1CB",
    label=f"Tabular only ({m_tab_mean:.3f})",
    linewidth=1.5,
)
ax.plot(
    x,
    m_wf_folds,
    "s-",
    color="#7F77DD",
    label=f"Waveform only ({m_wf_mean:.3f})",
    linewidth=1.5,
)
ax.plot(
    x,
    m_fus_folds,
    "^-",
    color="#1D9E75",
    label=f"Fusion ({m_fus_mean:.3f})",
    linewidth=2,
)
ax.axhline(0.663, color="#B4B2A9", linestyle="--", linewidth=1, label="HFA-ICOS 0.663")
ax.set_xlabel("CV fold")
ax.set_ylabel("C-index")
ax.set_title("Per-fold performance", fontweight="500")
ax.legend(fontsize=8)
ax.set_ylim(0.3, 1.0)
ax.spines[["top", "right"]].set_visible(False)

# 2. Summary bar chart
ax = axes[1]
names = ["HFA-ICOS\nbaseline", "Tabular\nonly", "Waveform\nonly", "Fusion"]
means = [0.663, m_tab_mean, m_wf_mean, m_fus_mean]
stds = [0, m_tab_std, m_wf_std, m_fus_std]
colors = ["#B4B2A9", "#9FE1CB", "#AFA9EC", "#1D9E75"]
bars = ax.bar(names, means, color=colors, alpha=0.85, width=0.55)
for i, (mean, std) in enumerate(zip(means, stds)):
    if std > 0:
        ax.errorbar(
            i, mean, yerr=std, fmt="none", ecolor="#444441", capsize=5, linewidth=1.5
        )
    ax.text(
        i,
        mean + 0.007,
        f"{mean:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="500",
    )
ax.set_ylim(0.45, 0.95)
ax.set_ylabel("CV C-index")
ax.set_title("Model comparison\n(270-patient subset)", fontweight="500")
ax.spines[["top", "right"]].set_visible(False)

# 3. KM curves by fusion model risk tertile
ax = axes[2]
colors_km = {"low": "#1D9E75", "medium": "#EF9F27", "high": "#D85A30"}
for grp in ["low", "medium", "high"]:
    subset = df_fus_complete[df_fus_complete["risk_group"] == grp]
    kmf = KaplanMeierFitter()
    kmf.fit(subset["time"] / 365, subset["CTRCD"])
    n_grp = len(subset)
    ev_grp = int(subset["CTRCD"].sum())
    ax.plot(
        kmf.timeline,
        1 - kmf.survival_function_.values.flatten(),
        color=colors_km[grp],
        linewidth=2,
        label=f"{grp} risk (n={n_grp}, ev={ev_grp})",
    )
ax.set_xlabel("Years")
ax.set_ylabel("Cumulative CTRCD incidence")
ax.set_title("KM curves by fusion model\nrisk tertile", fontweight="500")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
outpath = OUT / "fusion_model_results.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved → {outpath}")

# ── Final project summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PROJECT SUMMARY — current state")
print("=" * 60)
print(f"  HFA-ICOS clinical score (baseline):     0.663")
print(f"  Cox tabular model (full 479 patients):  0.712 ± 0.113")
print(f"  Cox + ACprev + RTprev (Model B):        0.743 ± 0.065")
print(f"  TDI waveform features only:             {m_wf_mean:.3f} ± {m_wf_std:.3f}")
print(f"  Fusion model (tabular + waveform):      {m_fus_mean:.3f} ± {m_fus_std:.3f}")
print(f"\nKey biological finding:")
print(f"  e' wave (diastolic velocity) is reduced in CTRCD patients")
print(f"  at baseline — subclinical diastolic dysfunction before")
print(f"  chemotherapy starts. Not captured by LVEF or HFA-ICOS.")
print(f"\nNext steps:")
print(f"  - Script 06: calibration analysis (do predicted probabilities")
print(f"    match observed rates?)")
print(f"  - Script 07: clinical utility — net benefit / decision curves")
print(f"  - Manuscript draft")
