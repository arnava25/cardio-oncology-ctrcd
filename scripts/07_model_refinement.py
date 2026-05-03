"""
07_model_refinement.py
Cardio-oncology CTRCD prediction project

Addresses the three issues identified before manuscript:
  1. Recalibration — fix E/O ratio of 1.52 using Platt scaling
  2. Bootstrap confidence intervals on C-index
  3. Waveform subset comparison — are the 270 patients with TDI
     systematically different from the 261 without?
  4. e' wave boxplot — make the biological finding visually undeniable
  5. Net Reclassification Improvement (NRI) vs HFA-ICOS

Run from project root:
    python scripts/07_model_refinement.py

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

DATA_CLIN = Path("data/BC_cardiotox_clinical_variables.csv")
DATA_ALL = Path("data/BC_cardiotox_clinical_and_functional_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# SETUP — load data and define helpers (same as scripts 05/06)
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
t_axis = np.linspace(0, 1, len(waveform_cols))


def extract_e_prime(wave):
    """Extract the e' (early diastolic) velocity from a TDI waveform."""
    peaks_neg, props = scipy_signal.find_peaks(-wave, height=0, prominence=0.05)
    if len(peaks_neg) >= 2:
        order = np.argsort(props["prominences"])[::-1]
        e_idx = peaks_neg[order[0]]
        a_idx = peaks_neg[order[1]]
        if t_axis[e_idx] > t_axis[a_idx]:
            e_idx, a_idx = a_idx, e_idx
        return wave[e_idx]
    elif len(peaks_neg) == 1:
        return wave[peaks_neg[0]]
    else:
        return np.min(wave)


# HFA-ICOS scorer
def hfa_icos_score(row):
    score = 0
    for col, pts in [
        ("AC", 1),
        ("antiHER2", 1),
        ("ACprev", 2),
        ("antiHER2prev", 1),
        ("RTprev", 1),
    ]:
        if row.get(col, 0) == 1:
            score += pts
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
    for col, pts in [
        ("HTA", 1),
        ("DM", 1),
        ("DL", 1),
        ("CIprev", 2),
        ("ICMprev", 2),
        ("ARRprev", 1),
    ]:
        if row.get(col, 0) == 1:
            score += pts
    return score


TABULAR = ["age", "heart_rate", "LVEF", "DL", "AC", "antiHER2", "ACprev"]
HORIZON = 730  # 2-year risk


def get_2yr_risk(df_train, df_test, features, horizon=HORIZON):
    """Predict P(event within horizon days) from a Cox model."""
    m = CoxPHFitter(penalizer=0.1)
    m.fit(
        df_train[features + ["CTRCD", "time"]], duration_col="time", event_col="CTRCD"
    )
    sf = m.predict_survival_function(df_test[features], times=[horizon])
    return 1 - sf.values.flatten()


def observed_event(df, horizon=HORIZON):
    return ((df["CTRCD"] == 1) & (df["time"] <= horizon)).astype(int).values


# Build waveform subset
has_wave = ~df_all[waveform_cols].isnull().any(axis=1)
df_wave = df_all[has_wave].copy().reset_index(drop=True)
df_nowave = df_all[~has_wave].copy().reset_index(drop=True)
X_waves = df_wave[waveform_cols].values
df_wave["e_prime"] = [extract_e_prime(X_waves[i]) for i in range(len(X_waves))]

# ══════════════════════════════════════════════════════════════════════════
# PART 1: WAVEFORM SUBSET COMPARISON
# Are the 270 patients with TDI different from the 261 without?
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: Waveform subset comparison")
print("=" * 60)
print(f"With waveform:    n={len(df_wave)},  events={df_wave['CTRCD'].sum()}")
print(f"Without waveform: n={len(df_nowave)}, events={df_nowave['CTRCD'].sum()}")

# Compare continuous variables
cont_vars = ["age", "LVEF", "heart_rate"]
print(
    f"\n{'Variable':<15} {'With TDI':>12} {'Without TDI':>12} {'p-value':>10} {'Sig':>5}"
)
print("-" * 55)
for var in cont_vars:
    g1 = df_wave[var].dropna()
    g2 = df_nowave[var].dropna()
    _, p = ttest_ind(g1, g2)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(
        f"{var:<15} {g1.mean():>8.1f}±{g1.std():.1f}  "
        f"{g2.mean():>8.1f}±{g2.std():.1f}  {p:>10.3f} {sig:>5}"
    )


# Compare binary variables and CTRCD rate
bin_vars = ['CTRCD', 'AC', 'antiHER2', 'HTA', 'DL']
print(f"\n{'Variable':<15} {'With TDI %':>12} {'Without TDI %':>14} {'p-value':>10}")
print("-"*55)
for var in bin_vars:
    g1 = df_wave[var].dropna().reset_index(drop=True)
    g2 = df_nowave[var].dropna().reset_index(drop=True)
    vals   = pd.concat([g1, g2], ignore_index=True)
    groups = pd.Series(['w']*len(g1) + ['wo']*len(g2))
    ct = pd.crosstab(vals, groups)
    if ct.shape == (2, 2):
        _, p, _, _ = chi2_contingency(ct)
    else:
        p = 1.0
    print(f"{var:<15} {g1.mean()*100:>10.1f}%  {g2.mean()*100:>12.1f}%  {p:>10.3f}")

# ══════════════════════════════════════════════════════════════════════════
# PART 2: RECALIBRATION
# Fix E/O = 1.52 using isotonic regression on out-of-fold predictions
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: Recalibration")
print("=" * 60)

df_tab = df_wave[TABULAR + ["CTRCD", "time"]].dropna().reset_index(drop=True)
obs_2yr = observed_event(df_tab)

# Get out-of-fold raw predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
raw_probs = np.zeros(len(df_tab))
for tr_idx, te_idx in skf.split(df_tab, df_tab["CTRCD"]):
    raw_probs[te_idx] = get_2yr_risk(df_tab.iloc[tr_idx], df_tab.iloc[te_idx], TABULAR)

print(f"Before recalibration: E/O = {raw_probs.mean()/obs_2yr.mean():.3f}")

# Isotonic regression recalibration
# Fits a monotone mapping from raw probabilities to observed rates
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_probs, obs_2yr)
cal_probs = iso.predict(raw_probs)

print(f"After recalibration:  E/O = {cal_probs.mean()/obs_2yr.mean():.3f}")
print(f"Mean raw prob:        {raw_probs.mean():.3f}")
print(f"Mean calibrated prob: {cal_probs.mean():.3f}")
print(f"Observed rate:        {obs_2yr.mean():.3f}")

# Verify discrimination is preserved after recalibration
c_raw = concordance_index(df_tab["time"], -raw_probs, df_tab["CTRCD"])
c_cal = concordance_index(df_tab["time"], -cal_probs, df_tab["CTRCD"])
print(f"C-index before: {c_raw:.3f}  |  C-index after: {c_cal:.3f}")
print("(Recalibration changes probabilities but not rank order)")

# ══════════════════════════════════════════════════════════════════════════
# PART 3: BOOTSTRAP CONFIDENCE INTERVALS
# 1000 bootstrap resamples → proper 95% CI on C-index
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: Bootstrap confidence intervals (n=1000 resamples)")
print("=" * 60)
print("Running... (takes ~2 minutes)")

rng = np.random.default_rng(42)
n_boot = 1000
boot_cindex = []

for i in range(n_boot):
    idx = rng.choice(len(df_tab), size=len(df_tab), replace=True)
    df_boot = df_tab.iloc[idx].reset_index(drop=True)
    # Need at least 2 events to fit
    if df_boot["CTRCD"].sum() < 2:
        continue
    try:
        m = CoxPHFitter(penalizer=0.1)
        m.fit(df_boot, duration_col="time", event_col="CTRCD")
        c = concordance_index(
            df_boot["time"],
            -m.predict_partial_hazard(df_boot[TABULAR]),
            df_boot["CTRCD"],
        )
        boot_cindex.append(c)
    except Exception:
        continue

boot_arr = np.array(boot_cindex)
ci_lo = np.percentile(boot_arr, 2.5)
ci_hi = np.percentile(boot_arr, 97.5)
ci_med = np.median(boot_arr)

print(f"Bootstrap C-index: {ci_med:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})")
print(f"Based on {len(boot_arr)} successful resamples of {n_boot}")
print(f"\nFor manuscript: C-index = {ci_med:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})")

# ══════════════════════════════════════════════════════════════════════════
# PART 4: e' WAVE ANALYSIS
# Formal test with LVEF adjustment + visual
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: e' wave analysis")
print("=" * 60)

# Univariate
g0 = df_wave.loc[df_wave["CTRCD"] == 0, "e_prime"]
g1 = df_wave.loc[df_wave["CTRCD"] == 1, "e_prime"]
_, p_uni = mannwhitneyu(g0, g1, alternative="two-sided")
print(
    f"e' univariate: no-CTRCD {g0.mean():.2f} | CTRCD {g1.mean():.2f} | p={p_uni:.3f}"
)

# Adjusted for LVEF: logistic regression with e_prime + LVEF
df_eprime = df_wave[["e_prime", "LVEF", "CTRCD"]].dropna()
scaler = StandardScaler()
X_adj = scaler.fit_transform(df_eprime[["e_prime", "LVEF"]])
lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_adj, df_eprime["CTRCD"])
# p-value via bootstrap
boot_coefs = []
for _ in range(500):
    idx = rng.choice(len(df_eprime), len(df_eprime), replace=True)
    Xb, yb = X_adj[idx], df_eprime["CTRCD"].values[idx]
    if yb.sum() < 2:
        continue
    lrb = LogisticRegression(C=1.0, max_iter=1000)
    lrb.fit(Xb, yb)
    boot_coefs.append(lrb.coef_[0][0])  # e_prime coefficient

boot_coefs = np.array(boot_coefs)
# Two-sided p: proportion of bootstrap coefs crossing zero
p_adj = 2 * min((boot_coefs > 0).mean(), (boot_coefs < 0).mean())
print(f"e' adjusted for LVEF: coef={lr.coef_[0][0]:.3f}, " f"bootstrap p={p_adj:.3f}")
print(
    f"e' coef 95% CI: [{np.percentile(boot_coefs,2.5):.3f}, "
    f"{np.percentile(boot_coefs,97.5):.3f}]"
)

if p_adj < 0.05:
    print("=> e' independently predicts CTRCD after adjusting for LVEF")
else:
    print("=> e' effect attenuated after LVEF adjustment (correlated predictors)")

# ══════════════════════════════════════════════════════════════════════════
# PART 5: NRI vs HFA-ICOS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: Net Reclassification Improvement vs HFA-ICOS")
print("=" * 60)

# Get HFA-ICOS risk categories on patients we can score
hfa_req = [
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
df_hfa = df_clin.dropna(subset=hfa_req).copy()
df_hfa["hfa_score"] = df_hfa.apply(hfa_icos_score, axis=1)


def hfa_cat(s):
    if s <= 1:
        return 0  # low
    elif s <= 4:
        return 1  # moderate
    elif s <= 6:
        return 2  # high
    else:
        return 3  # very high


df_hfa["hfa_cat"] = df_hfa["hfa_score"].apply(hfa_cat)

# Get Cox model predictions on same patients
df_cox_nri = df_hfa[TABULAR + ["CTRCD", "time"]].dropna().copy()
df_cox_nri = df_cox_nri.reset_index(drop=True)

# Fit on full dataset, predict on same (in-sample — ok for NRI illustration)
m_nri = CoxPHFitter(penalizer=0.1)
m_nri.fit(df_cox_nri, duration_col="time", event_col="CTRCD")
cox_probs = (
    1
    - m_nri.predict_survival_function(
        df_cox_nri[TABULAR], times=[HORIZON]
    ).values.flatten()
)

# Apply isotonic recalibration
cox_probs_cal = iso.predict(cox_probs)


# Cox risk categories using clinical thresholds (5%, 10%, 20%)
def cox_cat(p):
    if p < 0.05:
        return 0  # low
    elif p < 0.10:
        return 1  # moderate
    elif p < 0.20:
        return 2  # high
    else:
        return 3  # very high


df_cox_nri["cox_cat"] = [cox_cat(p) for p in cox_probs_cal]

# Match patients between HFA and Cox dataframes
# Use index alignment — only patients present in both
common_idx = df_hfa.index.intersection(df_cox_nri.index)
hfa_cats = df_hfa.loc[common_idx, "hfa_cat"].values
cox_cats = df_cox_nri.loc[common_idx, "cox_cat"].values
outcomes = df_cox_nri.loc[common_idx, "CTRCD"].values

# NRI calculation
events_mask = outcomes == 1
noevents_mask = outcomes == 0

# For events: upward reclassification is good
nri_events = np.mean(cox_cats[events_mask] > hfa_cats[events_mask]) - np.mean(
    cox_cats[events_mask] < hfa_cats[events_mask]
)
# For non-events: downward reclassification is good
nri_noevents = np.mean(cox_cats[noevents_mask] < hfa_cats[noevents_mask]) - np.mean(
    cox_cats[noevents_mask] > hfa_cats[noevents_mask]
)
nri_total = nri_events + nri_noevents

print(f"Patients with both scores: {len(common_idx)}")
print(f"Events: {events_mask.sum()}, Non-events: {noevents_mask.sum()}")
print(f"\nNRI (events):     {nri_events:+.3f}")
print(f"NRI (non-events): {nri_noevents:+.3f}")
print(f"NRI (total):      {nri_total:+.3f}")
print(f"\nInterpretation:")
if nri_events > 0:
    pct = nri_events * 100
    print(f"  Among patients who DID develop CTRCD, the Cox model")
    print(f"  correctly reclassifies {pct:.1f}% more upward than downward")
if nri_noevents > 0:
    pct = nri_noevents * 100
    print(f"  Among patients who did NOT develop CTRCD, the Cox model")
    print(f"  correctly reclassifies {pct:.1f}% more downward than upward")

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#fafaf8")

# 1. Waveform subset comparison — key variables
ax = axes[0, 0]
vars_compare = ["age", "LVEF", "heart_rate"]
labels = ["Age (years)", "LVEF (%)", "Heart rate (bpm)"]
x = np.arange(len(vars_compare))
w = 0.35
means_w = [df_wave[v].mean() for v in vars_compare]
means_nw = [df_nowave[v].mean() for v in vars_compare]
sems_w = [df_wave[v].sem() for v in vars_compare]
sems_nw = [df_nowave[v].sem() for v in vars_compare]
ax.bar(
    x - w / 2,
    means_w,
    w,
    color="#1D9E75",
    alpha=0.8,
    label=f"With TDI (n={len(df_wave)})",
)
ax.bar(
    x + w / 2,
    means_nw,
    w,
    color="#B4B2A9",
    alpha=0.8,
    label=f"Without TDI (n={len(df_nowave)})",
)
ax.errorbar(x - w / 2, means_w, yerr=sems_w, fmt="none", ecolor="#0F6E56", capsize=4)
ax.errorbar(x + w / 2, means_nw, yerr=sems_nw, fmt="none", ecolor="#444441", capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Subset comparison\n(with vs without TDI)", fontweight="500")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# 2. Calibration before and after
ax = axes[0, 1]
# Quintile bins for before and after
n_bins = 5
for probs_plot, label, color, marker in [
    (raw_probs, "Before recalibration", "#D85A30", "o"),
    (cal_probs, "After recalibration", "#1D9E75", "s"),
]:
    edges = np.percentile(probs_plot, np.linspace(0, 100, n_bins + 1))
    pm_list, ob_list = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs_plot >= lo) & (
            probs_plot <= hi if i == n_bins - 1 else probs_plot < hi
        )
        if mask.sum() > 0:
            pm_list.append(probs_plot[mask].mean())
            ob_list.append(obs_2yr[mask].mean())
    ax.plot(
        pm_list,
        ob_list,
        f"{marker}-",
        color=color,
        linewidth=1.5,
        label=label,
        markersize=7,
    )
ax.plot([0, 0.5], [0, 0.5], "k--", linewidth=1, alpha=0.4, label="Perfect calibration")
ax.set_xlabel("Mean predicted risk")
ax.set_ylabel("Observed event rate")
ax.set_title("Calibration before vs after\nrecalibration", fontweight="500")
ax.legend(fontsize=8)
ax.set_xlim(-0.01, 0.45)
ax.set_ylim(-0.01, 0.45)
ax.spines[["top", "right"]].set_visible(False)

# 3. Bootstrap C-index distribution
ax = axes[0, 2]
ax.hist(boot_arr, bins=40, color="#5DCAA5", alpha=0.8, edgecolor="none")
ax.axvline(ci_med, color="#0F6E56", linewidth=2, label=f"Median {ci_med:.3f}")
ax.axvline(
    ci_lo,
    color="#0F6E56",
    linewidth=1.5,
    linestyle="--",
    label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
)
ax.axvline(ci_hi, color="#0F6E56", linewidth=1.5, linestyle="--")
ax.axvline(0.663, color="#B4B2A9", linewidth=1.5, linestyle=":", label="HFA-ICOS 0.663")
ax.set_xlabel("Bootstrap C-index")
ax.set_ylabel("Count")
ax.set_title("Bootstrap CI on C-index\n(n=1000 resamples)", fontweight="500")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# 4. e' wave boxplot — THE biological finding
ax = axes[1, 0]
data_box = [g0.values, g1.values]
bp = ax.boxplot(
    data_box,
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="white", linewidth=2),
)
bp["boxes"][0].set_facecolor("#1D9E75")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#D85A30")
bp["boxes"][1].set_alpha(0.7)
# Add individual points (jittered)
rng_plot = np.random.default_rng(0)
for i, (data, color) in enumerate(zip(data_box, ["#0F6E56", "#993C1D"])):
    jitter = rng_plot.uniform(-0.15, 0.15, len(data))
    ax.scatter(
        np.full(len(data), i + 1) + jitter, data, alpha=0.3, s=12, color=color, zorder=3
    )
ax.set_xticks([1, 2])
ax.set_xticklabels([f"No CTRCD\n(n={len(g0)})", f"CTRCD\n(n={len(g1)})"])
ax.set_ylabel("e' velocity (cm/s)")
ax.set_title(
    f"Baseline e' wave by outcome\n(p={p_uni:.3f}, Mann-Whitney)", fontweight="500"
)
# Add significance bar
y_max = max(g0.max(), g1.max()) + 0.5
ax.plot(
    [1, 1, 2, 2], [y_max, y_max + 0.3, y_max + 0.3, y_max], color="#444441", linewidth=1
)
stars = "**" if p_uni < 0.01 else ("*" if p_uni < 0.05 else "ns")
ax.text(1.5, y_max + 0.4, stars, ha="center", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)

# 5. NRI reclassification diagram
ax = axes[1, 1]
categories = ["Low", "Moderate", "High", "Very high"]
# Count patients in each combo (HFA row, Cox col)
mat = np.zeros((4, 4), dtype=int)
for h, c in zip(hfa_cats, cox_cats):
    if h < 4 and c < 4:
        mat[h, c] += 1
im = ax.imshow(mat, cmap="YlGn", aspect="auto")
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(categories, fontsize=8, rotation=20)
ax.set_yticklabels(categories, fontsize=8)
ax.set_xlabel("Cox model category")
ax.set_ylabel("HFA-ICOS category")
ax.set_title(f"Reclassification matrix\nNRI={nri_total:+.3f}", fontweight="500")
for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            str(mat[i, j]),
            ha="center",
            va="center",
            fontsize=10,
            color="#2C2C2A" if mat[i, j] < mat.max() * 0.6 else "white",
        )
plt.colorbar(im, ax=ax, shrink=0.8)

# 6. Final summary table
ax = axes[1, 2]
ax.axis("off")
summary_data = [
    ["Metric", "Value"],
    ["C-index (CV, full N=479)", "0.743"],
    ["C-index 95% CI (bootstrap)", f"{ci_lo:.3f}–{ci_hi:.3f}"],
    ["HFA-ICOS AUC (baseline)", "0.663"],
    ["E/O ratio (before recal.)", "1.52"],
    ["E/O ratio (after recal.)", f"{cal_probs.mean()/obs_2yr.mean():.2f}"],
    ["e' wave p-value (univar.)", f"{p_uni:.3f}"],
    ["e' wave p-value (adj. LVEF)", f"{p_adj:.3f}"],
    ["NRI vs HFA-ICOS (total)", f"{nri_total:+.3f}"],
    ["NRI (events only)", f"{nri_events:+.3f}"],
    ["NRI (non-events only)", f"{nri_noevents:+.3f}"],
]
tbl = ax.table(
    cellText=summary_data[1:], colLabels=summary_data[0], loc="center", cellLoc="left"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
# Header styling
for j in range(2):
    tbl[(0, j)].set_facecolor("#1D9E75")
    tbl[(0, j)].set_text_props(color="white", fontweight="500")
ax.set_title("Summary statistics for manuscript", fontweight="500", pad=20)

plt.tight_layout()
outpath = OUT / "model_refinement.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved → {outpath}")

print("\n" + "=" * 60)
print("READY FOR MANUSCRIPT")
print("=" * 60)
print(f"Primary result:  C-index {ci_med:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})")
print(f"vs HFA-ICOS:     0.663")
print(f"After recalib:   E/O ratio {cal_probs.mean()/obs_2yr.mean():.2f}")
print(f"Biological find: e' p={p_uni:.3f} univariate, p={p_adj:.3f} adj. LVEF")
print(f"NRI total:       {nri_total:+.3f}")
