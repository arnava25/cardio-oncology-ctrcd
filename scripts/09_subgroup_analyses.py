"""
09_subgroup_analyses.py — Subgroup analyses
Cardio-oncology CTRCD prediction project

Tests whether the Cox model performs consistently across clinically
meaningful patient subgroups. Two types of analysis:

  1. DESCRIPTIVE SUBGROUPS — CTRCD rates and model performance
     stratified by treatment type and age group

  2. INTERACTION TESTS — formal statistical test for whether the
     model's predictive variables have different effects in different
     subgroups (effect modification). A significant interaction
     means the model needs to be adjusted for that subgroup;
     a non-significant interaction supports uniform application.

Subgroups tested:
  - Treatment: AC only vs antiHER2 only vs both
  - Age: <50, 50-65, >65
  - Baseline LVEF: <60%, 60-70%, >70%
  - Cardiovascular risk: low (0 risk factors) vs high (≥1 risk factor)

Run from project root:
    python scripts/09_subgroup_analyses.py

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
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2_contingency, fisher_exact
import warnings

warnings.filterwarnings("ignore")

DATA = Path("data/BC_cardiotox_clinical_variables.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# LOAD AND PREPARE
# ══════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA, sep=";", decimal=",")
print(f"Loaded {len(df)} patients, {df['CTRCD'].sum()} CTRCD events")

TABULAR = ["age", "heart_rate", "LVEF", "DL", "AC", "antiHER2", "ACprev"]

# ── Define subgroups ───────────────────────────────────────────────────────

# Treatment subgroups
df["tx_group"] = "unknown"
df.loc[(df["AC"] == 1) & (df["antiHER2"] == 0), "tx_group"] = "AC only"
df.loc[(df["AC"] == 0) & (df["antiHER2"] == 1), "tx_group"] = "antiHER2 only"
df.loc[(df["AC"] == 1) & (df["antiHER2"] == 1), "tx_group"] = "AC + antiHER2"
df.loc[(df["AC"] == 0) & (df["antiHER2"] == 0), "tx_group"] = "neither"

# Age subgroups
df["age_group"] = pd.cut(
    df["age"], bins=[0, 50, 65, 120], labels=["<50", "50-65", ">65"]
)

# LVEF subgroups
df["lvef_group"] = pd.cut(
    df["LVEF"], bins=[0, 60, 70, 100], labels=["<60%", "60-70%", ">70%"]
)

# CV risk score (number of risk factors: HTA, DL, DM, smoker)
risk_cols = ["HTA", "DL", "DM", "smoker"]
df["cv_risk_count"] = df[risk_cols].sum(axis=1)
df["cv_risk_group"] = (df["cv_risk_count"] >= 1).map(
    {True: "≥1 CV risk factor", False: "No CV risk factors"}
)

# ══════════════════════════════════════════════════════════════════════════
# PART 1: DESCRIPTIVE SUBGROUP ANALYSIS
# CTRCD rates and unadjusted event rates per subgroup
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 1: CTRCD rates by subgroup")
print("=" * 60)


def subgroup_summary(df_sub, group_col, min_events=3):
    """Print CTRCD rate and event count for each level of group_col."""
    groups = df_sub.groupby(group_col, observed=True)
    rows = []
    for name, grp in groups:
        n = len(grp)
        events = int(grp["CTRCD"].sum())
        rate = grp["CTRCD"].mean() * 100
        median_fu = grp["time"].median() / 365
        rows.append(
            {
                "Subgroup": name,
                "N": n,
                "Events": events,
                "CTRCD %": f"{rate:.1f}%",
                "Median FU (yr)": f"{median_fu:.1f}",
            }
        )
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


print("\n--- Treatment subgroups ---")
subgroup_summary(df.dropna(subset=["AC", "antiHER2"]), "tx_group")

print("\n--- Age subgroups ---")
subgroup_summary(df, "age_group")

print("\n--- LVEF subgroups ---")
subgroup_summary(df.dropna(subset=["LVEF"]), "lvef_group")

print("\n--- CV risk subgroups ---")
subgroup_summary(df.dropna(subset=risk_cols), "cv_risk_group")

# ══════════════════════════════════════════════════════════════════════════
# PART 2: MODEL PERFORMANCE BY SUBGROUP
# C-index within each subgroup — does the model work equally well?
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: Model C-index by subgroup")
print("=" * 60)
print("(Fitting on full dataset, evaluating C-index within each subgroup)")
print("Note: small subgroups give unstable C-index estimates\n")

# Fit model on complete cases
df_complete = df[TABULAR + ["CTRCD", "time"]].dropna()
cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_complete, duration_col="time", event_col="CTRCD")

# Add predicted risk score back to full df
df_scored = df.copy()
score_idx = df_complete.index
df_scored.loc[score_idx, "risk_score"] = cph.predict_partial_hazard(
    df_complete[TABULAR]
).values


def subgroup_cindex(df_sub, group_col, min_events=4):
    """C-index within each subgroup level."""
    groups = df_sub.dropna(subset=["risk_score"]).groupby(group_col, observed=True)
    print(f"{'Subgroup':<22} {'N':>5} {'Events':>7} {'C-index':>9}")
    print("-" * 45)
    for name, grp in groups:
        grp_clean = grp.dropna(subset=["risk_score", "CTRCD", "time"])
        n = len(grp_clean)
        events = int(grp_clean["CTRCD"].sum())
        if events < min_events:
            print(f"  {str(name):<20} {n:>5} {events:>7}   {'<4 events':>9}")
            continue
        c = concordance_index(
            grp_clean["time"], -grp_clean["risk_score"], grp_clean["CTRCD"]
        )
        print(f"  {str(name):<20} {n:>5} {events:>7} {c:>9.3f}")


print("\n--- Treatment subgroups ---")
subgroup_cindex(df_scored.dropna(subset=["AC", "antiHER2"]), "tx_group")

print("\n--- Age subgroups ---")
subgroup_cindex(df_scored, "age_group")

print("\n--- LVEF subgroups ---")
subgroup_cindex(df_scored.dropna(subset=["LVEF"]), "lvef_group")

print("\n--- CV risk subgroups ---")
subgroup_cindex(df_scored.dropna(subset=risk_cols), "cv_risk_group")

# ══════════════════════════════════════════════════════════════════════════
# PART 3: INTERACTION TESTS
# Does the effect of each predictor differ significantly by subgroup?
# Test: add interaction term to Cox model, check p-value
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: Interaction tests (effect modification)")
print("=" * 60)
print("H0: predictor effect is the same across subgroups")
print("p < 0.05 = significant interaction (effect modifier)\n")


def test_interaction(
    df_test, predictor, modifier_col, modifier_binary=True, penalizer=0.1
):
    """
    Test whether 'modifier_col' significantly modifies the effect
    of 'predictor' on CTRCD time-to-event outcome.
    Returns p-value of the interaction term.
    """
    needed = list(dict.fromkeys(
        [predictor, modifier_col, "CTRCD", "time"] +
        [v for v in TABULAR if v != predictor]
    ))
    df_int = df_test[needed].dropna()

    if modifier_binary:
        # Binary modifier: create dummy
        mod_vals = df_int[modifier_col].squeeze().unique()
        if len(mod_vals) > 2:
            return None, None
        ref_val = sorted(mod_vals)[0]
        df_int = df_int.copy()
        df_int["mod_dummy"] = (df_int[modifier_col] != ref_val).astype(int)
        df_int["interaction"] = df_int[predictor] * df_int["mod_dummy"]
        fit_cols = [v for v in TABULAR if v != predictor] + [
            predictor,
            "mod_dummy",
            "interaction",
        ]
    else:
        df_int = df_int.copy()
        df_int["interaction"] = df_int[predictor] * df_int[modifier_col]
        fit_cols = TABULAR + ["interaction"]

    # Remove duplicate columns
    fit_cols = list(dict.fromkeys([c for c in fit_cols if c in df_int.columns]))

    try:
        m = CoxPHFitter(penalizer=penalizer)
        m.fit(
            df_int[fit_cols + ["CTRCD", "time"]], duration_col="time", event_col="CTRCD"
        )
        p = m.summary.loc["interaction", "p"]
        hr = m.summary.loc["interaction", "exp(coef)"]
        return p, hr
    except Exception as e:
        return None, None


# Key interactions to test
interactions = [
    # (predictor, modifier, description)
    ("age", "AC", "Age × anthracyclines"),
    ("age", "antiHER2", "Age × anti-HER2"),
    ("LVEF", "AC", "LVEF × anthracyclines"),
    ("ACprev", "AC", "Prior AC × current AC"),
    ("heart_rate", "HTA", "Heart rate × hypertension"),
]

print(f"{'Interaction':<35} {'HR':>6} {'p-value':>10} {'Sig':>5}")
print("-" * 58)
for pred, mod, desc in interactions:
    df_int_test = df.dropna(subset=TABULAR + [mod])
    p, hr = test_interaction(df_int_test, pred, mod)
    if p is not None:
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.1 else "ns"))
        print(f"  {desc:<33} {hr:>6.3f} {p:>10.3f} {sig:>5}")
    else:
        print(f"  {desc:<33} {'--':>6} {'--':>10} {'--':>5}")

# ══════════════════════════════════════════════════════════════════════════
# PART 4: FOREST PLOT — treatment subgroup C-indices
# Standard clinical paper presentation
# ══════════════════════════════════════════════════════════════════════════

# Collect all subgroup C-indices for forest plot
forest_data = []

subgroup_defs = [
    ("Age <50", df_scored["age_group"] == "<50"),
    ("Age 50-65", df_scored["age_group"] == "50-65"),
    ("Age >65", df_scored["age_group"] == ">65"),
    ("AC only", (df_scored["tx_group"] == "AC only") & df_scored["risk_score"].notna()),
    (
        "antiHER2 only",
        (df_scored["tx_group"] == "antiHER2 only") & df_scored["risk_score"].notna(),
    ),
    (
        "AC + antiHER2",
        (df_scored["tx_group"] == "AC + antiHER2") & df_scored["risk_score"].notna(),
    ),
    ("LVEF <60%", df_scored["lvef_group"] == "<60%"),
    ("LVEF 60-70%", df_scored["lvef_group"] == "60-70%"),
    ("LVEF >70%", df_scored["lvef_group"] == ">70%"),
    ("No CV risk", df_scored["cv_risk_group"] == "No CV risk factors"),
    ("≥1 CV risk", df_scored["cv_risk_group"] == "≥1 CV risk factor"),
]

for label, mask in subgroup_defs:
    grp = df_scored[mask].dropna(subset=["risk_score", "CTRCD", "time"])
    n = len(grp)
    events = int(grp["CTRCD"].sum())
    if events < 4:
        forest_data.append(
            {
                "label": label,
                "n": n,
                "events": events,
                "c": None,
                "ci_lo": None,
                "ci_hi": None,
            }
        )
        continue
    # Bootstrap CI for each subgroup C-index
    c_obs = concordance_index(grp["time"], -grp["risk_score"], grp["CTRCD"])
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(500):
        idx = rng.choice(len(grp), len(grp), replace=True)
        g = grp.iloc[idx]
        if g["CTRCD"].sum() < 2:
            continue
        try:
            boots.append(concordance_index(g["time"], -g["risk_score"], g["CTRCD"]))
        except:
            pass
    if boots:
        ci_lo = np.percentile(boots, 2.5)
        ci_hi = np.percentile(boots, 97.5)
    else:
        ci_lo = ci_hi = c_obs
    forest_data.append(
        {
            "label": label,
            "n": n,
            "events": events,
            "c": c_obs,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    )

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#fafaf8")

# Layout: left = forest plot, right column = KM curves
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)

# ── Forest plot ────────────────────────────────────────────────────────────
ax_forest = fig.add_subplot(gs[:, 0])

valid = [d for d in forest_data if d["c"] is not None]
invalid = [d for d in forest_data if d["c"] is None]
y_pos = np.arange(len(forest_data))

# Overall model C-index
overall_c = concordance_index(
    df_complete["time"],
    -cph.predict_partial_hazard(df_complete[TABULAR]),
    df_complete["CTRCD"],
)

ax_forest.axvline(
    overall_c,
    color="#888780",
    linestyle="--",
    linewidth=1,
    label=f"Overall C-index {overall_c:.3f}",
)
ax_forest.axvline(
    0.5, color="#D3D1C7", linestyle=":", linewidth=0.8, label="No discrimination (0.5)"
)

for i, d in enumerate(forest_data):
    y = len(forest_data) - 1 - i
    if d["c"] is not None:
        ax_forest.errorbar(
            d["c"],
            y,
            xerr=[[d["c"] - d["ci_lo"]], [d["ci_hi"] - d["c"]]],
            fmt="o",
            color="#1D9E75",
            ecolor="#9FE1CB",
            capsize=4,
            markersize=7,
            linewidth=1.5,
        )
        label_txt = f"{d['label']}"
        stat_txt = f"n={d['n']}, ev={d['events']}, C={d['c']:.2f}"
    else:
        ax_forest.plot(0.5, y, "x", color="#B4B2A9", markersize=8)
        label_txt = f"{d['label']}"
        stat_txt = f"n={d['n']}, ev={d['events']} (too few)"
    ax_forest.text(
        -0.02,
        y,
        label_txt,
        ha="right",
        va="center",
        fontsize=9,
        transform=ax_forest.get_yaxis_transform(),
    )
    ax_forest.text(
        1.02,
        y,
        stat_txt,
        ha="left",
        va="center",
        fontsize=7.5,
        color="#5F5E5A",
        transform=ax_forest.get_yaxis_transform(),
    )

ax_forest.set_xlim(0.3, 1.05)
ax_forest.set_ylim(-0.5, len(forest_data) - 0.5)
ax_forest.set_yticks([])
ax_forest.set_xlabel("C-index (95% bootstrap CI)")
ax_forest.set_title("Model C-index by subgroup\n(forest plot)", fontweight="500")
ax_forest.legend(fontsize=8, loc="lower right")
ax_forest.spines[["top", "right", "left"]].set_visible(False)

# ── KM curves: treatment subgroups ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
tx_colors = {
    "AC only": "#EF9F27",
    "antiHER2 only": "#1D9E75",
    "AC + antiHER2": "#D85A30",
    "neither": "#B4B2A9",
}
df_tx = df.dropna(subset=["AC", "antiHER2"])
for grp_name, color in tx_colors.items():
    subset = df_tx[df_tx["tx_group"] == grp_name]
    if subset["CTRCD"].sum() < 2:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(subset["time"] / 365, subset["CTRCD"])
    n_ev = int(subset["CTRCD"].sum())
    ax1.plot(
        kmf.timeline,
        1 - kmf.survival_function_.values.flatten(),
        color=color,
        linewidth=1.8,
        label=f"{grp_name} (n={len(subset)}, ev={n_ev})",
    )
ax1.set_xlabel("Years")
ax1.set_ylabel("Cumulative CTRCD")
ax1.set_title("By treatment type", fontweight="500")
ax1.legend(fontsize=7)
ax1.set_xlim(0, 12)
ax1.spines[["top", "right"]].set_visible(False)

# ── KM curves: age subgroups ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
age_colors = {"<50": "#1D9E75", "50-65": "#EF9F27", ">65": "#D85A30"}
for grp_name, color in age_colors.items():
    subset = df[df["age_group"] == grp_name]
    if subset["CTRCD"].sum() < 2:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(subset["time"] / 365, subset["CTRCD"])
    n_ev = int(subset["CTRCD"].sum())
    ax2.plot(
        kmf.timeline,
        1 - kmf.survival_function_.values.flatten(),
        color=color,
        linewidth=1.8,
        label=f"Age {grp_name} (n={len(subset)}, ev={n_ev})",
    )
ax2.set_xlabel("Years")
ax2.set_ylabel("Cumulative CTRCD")
ax2.set_title("By age group", fontweight="500")
ax2.legend(fontsize=8)
ax2.set_xlim(0, 12)
ax2.spines[["top", "right"]].set_visible(False)

# ── KM curves: LVEF subgroups ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 1])
lvef_colors = {"<60%": "#D85A30", "60-70%": "#EF9F27", ">70%": "#1D9E75"}
for grp_name, color in lvef_colors.items():
    subset = df.dropna(subset=["LVEF"])[
        df.dropna(subset=["LVEF"])["lvef_group"] == grp_name
    ]
    if subset["CTRCD"].sum() < 2:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(subset["time"] / 365, subset["CTRCD"])
    n_ev = int(subset["CTRCD"].sum())
    ax3.plot(
        kmf.timeline,
        1 - kmf.survival_function_.values.flatten(),
        color=color,
        linewidth=1.8,
        label=f"LVEF {grp_name} (n={len(subset)}, ev={n_ev})",
    )
ax3.set_xlabel("Years")
ax3.set_ylabel("Cumulative CTRCD")
ax3.set_title("By baseline LVEF", fontweight="500")
ax3.legend(fontsize=8)
ax3.set_xlim(0, 12)
ax3.spines[["top", "right"]].set_visible(False)

plt.suptitle("Subgroup analyses", fontsize=14, fontweight="500", y=1.01)
outpath = OUT / "subgroup_analyses.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#fafaf8")
print(f"\nSaved → {outpath}")

# ── Manuscript summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUBGROUP ANALYSES — MANUSCRIPT SUMMARY")
print("=" * 60)
print("\nForest plot summary:")
for d in forest_data:
    if d["c"] is not None:
        print(
            f"  {d['label']:<22}: C={d['c']:.3f} "
            f"(95% CI {d['ci_lo']:.3f}–{d['ci_hi']:.3f}), "
            f"n={d['n']}, events={d['events']}"
        )
    else:
        print(f"  {d['label']:<22}: insufficient events (n={d['n']}, ev={d['events']})")
