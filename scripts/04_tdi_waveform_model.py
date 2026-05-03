"""
04_tdi_waveform_model.py — TDI waveform analysis + 1D CNN
Cardio-oncology CTRCD prediction project

The TDI (Tissue Doppler Imaging) waveform measures myocardial wall velocity
during each cardiac cycle. Each patient has a 1001-point time series
extracted from their baseline echocardiogram.

This script does three things:
  1. Explores the waveform data — visualizes average waveforms by outcome
  2. Extracts hand-crafted features from the waveform (peaks, troughs,
     timing, area under curve) — these are interpretable and work with
     small samples
  3. Trains a 1D CNN on the raw waveform — more powerful but needs care
     at this sample size

Why this matters: no existing clinical risk score uses the TDI waveform
as a predictive feature. It encodes subtle pre-existing myocardial
dysfunction that LVEF alone misses.

Run from project root:
    python scripts/04_tdi_waveform_model.py

Dependencies:
    pip install torch  (for the CNN section)

Outputs saved to: results/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
DATA_FUNC = Path('data/BC_cardiotox_functional_variable.csv')
DATA_CLIN = Path('data/BC_cardiotox_clinical_variables.csv')
OUT       = Path('results')
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# PART 1: Load and explore the waveform data
# ══════════════════════════════════════════════════════════════════════════

print("=== Loading TDI waveform data ===")
df_func = pd.read_csv(DATA_FUNC, sep=';', decimal=',')
print(f"Shape: {df_func.shape}")
print(f"Columns: CTRCD + {df_func.shape[1]-1} timepoints")
print(f"Patients: {len(df_func)}")
print(f"CTRCD distribution:\n{df_func['CTRCD'].value_counts()}")
print(f"CTRCD rate: {df_func['CTRCD'].mean()*100:.1f}%")

# Extract waveform matrix (1001 timepoints per patient)
waveform_cols = [c for c in df_func.columns if c.startswith('t ') or c.startswith('t')]
waveform_cols = sorted(waveform_cols, key=lambda x: int(x.replace('t ', '').replace('t', '').strip()))
print(f"\nWaveform columns found: {len(waveform_cols)}")

X_waves = df_func[waveform_cols].values  # shape: (270, 1001)
y_ctrcd = df_func['CTRCD'].values

# Check for any remaining NaNs in waveforms
nan_mask = np.any(np.isnan(X_waves), axis=1)
print(f"Patients with complete waveforms: {(~nan_mask).sum()} of {len(X_waves)}")

# Work only with complete waveforms
X_clean = X_waves[~nan_mask]
y_clean = y_ctrcd[~nan_mask]
print(f"Working dataset: {len(X_clean)} patients, {y_clean.sum()} events")

# Time axis: waveform is discretized over [0, 1] in 1001 points
t = np.linspace(0, 1, 1001)

# ── PART 1a: Visualize average waveforms by outcome ────────────────────────
print("\n=== Plotting average waveforms by outcome ===")
mean_no_ctrcd = X_clean[y_clean == 0].mean(axis=0)
mean_ctrcd    = X_clean[y_clean == 1].mean(axis=0)
std_no_ctrcd  = X_clean[y_clean == 0].std(axis=0)
std_ctrcd     = X_clean[y_clean == 1].std(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#fafaf8')

ax = axes[0]
ax.plot(t, mean_no_ctrcd, color='#1D9E75', linewidth=2,
        label=f'No CTRCD (n={( y_clean==0).sum()})')
ax.fill_between(t,
                mean_no_ctrcd - std_no_ctrcd,
                mean_no_ctrcd + std_no_ctrcd,
                alpha=0.15, color='#1D9E75')
ax.plot(t, mean_ctrcd, color='#D85A30', linewidth=2,
        label=f'CTRCD (n={(y_clean==1).sum()})')
ax.fill_between(t,
                mean_ctrcd - std_ctrcd,
                mean_ctrcd + std_ctrcd,
                alpha=0.15, color='#D85A30')
ax.set_xlabel('Normalised cardiac cycle (0 → 1)')
ax.set_ylabel('Myocardial velocity (cm/s)')
ax.set_title('Mean TDI waveform by outcome', fontweight='500')
ax.legend()
ax.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

# Difference waveform — where in the cycle do the groups diverge most?
ax = axes[1]
diff = mean_ctrcd - mean_no_ctrcd
ax.plot(t, diff, color='#534AB7', linewidth=1.5)
ax.fill_between(t, diff, 0, where=(diff < 0), alpha=0.3,
                color='#D85A30', label='CTRCD lower velocity')
ax.fill_between(t, diff, 0, where=(diff > 0), alpha=0.3,
                color='#1D9E75', label='CTRCD higher velocity')
ax.axhline(0, color='#888780', linewidth=0.8)
ax.set_xlabel('Normalised cardiac cycle (0 → 1)')
ax.set_ylabel('Velocity difference (CTRCD − no CTRCD)')
ax.set_title('Waveform difference: where groups diverge', fontweight='500')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / 'tdi_waveform_exploration.png', dpi=150,
            bbox_inches='tight', facecolor='#fafaf8')
print("Saved → results/tdi_waveform_exploration.png")

# ══════════════════════════════════════════════════════════════════════════
# PART 2: Hand-crafted waveform features
# These are interpretable, clinically meaningful, and work at small N.
# Extract them from the raw waveform and use as inputs to a simple model.
# ══════════════════════════════════════════════════════════════════════════

def extract_waveform_features(wave):
    """
    Extract physiologically meaningful scalars from a TDI waveform.

    TDI waveform anatomy:
      - Systolic wave (S'): positive peak during ventricular contraction
      - Early diastolic wave (e'): negative trough — myocardial relaxation
      - Late diastolic wave (a'): second negative trough — atrial contraction
      - e'/a' ratio: marker of diastolic function (reduced = diastolic dysfunction)
    """
    features = {}

    # Basic statistics
    features['mean']      = np.mean(wave)
    features['std']       = np.std(wave)
    features['max']       = np.max(wave)           # peak positive velocity (S' wave)
    features['min']       = np.min(wave)           # peak negative velocity (e' wave)
    features['range']     = np.max(wave) - np.min(wave)

    # Area under curve — positive vs negative portions
    # np.trapezoid in numpy >= 2.0, np.trapz in older versions

    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz
    features['auc_pos']   = _trapz(np.clip(wave, 0, None), t)
    features['auc_neg']   = _trapz(np.clip(wave, None, 0), t)  # will be negative
    features['auc_ratio'] = features['auc_pos'] / (abs(features['auc_neg']) + 1e-6)
    
    # Peak locations (timing of key cardiac events)
    features['t_max']     = t[np.argmax(wave)]     # time of peak systolic velocity
    features['t_min']     = t[np.argmin(wave)]     # time of peak e' velocity

    # Systolic peak prominence (S' wave)
    peaks_pos, props_pos = scipy_signal.find_peaks(wave, height=0, prominence=0.1)
    if len(peaks_pos) > 0:
        best_peak = peaks_pos[np.argmax(props_pos['prominences'])]
        features['s_prime']     = wave[best_peak]
        features['s_prime_t']   = t[best_peak]
    else:
        features['s_prime']     = features['max']
        features['s_prime_t']   = features['t_max']

    # Diastolic troughs (e' and a' waves — negative peaks)
    peaks_neg, props_neg = scipy_signal.find_peaks(-wave, height=0, prominence=0.05)
    if len(peaks_neg) >= 2:
        # Sort by prominence descending
        order = np.argsort(props_neg['prominences'])[::-1]
        e_prime_idx = peaks_neg[order[0]]
        a_prime_idx = peaks_neg[order[1]]
        # e' is the earlier prominent negative peak
        if t[e_prime_idx] > t[a_prime_idx]:
            e_prime_idx, a_prime_idx = a_prime_idx, e_prime_idx
        features['e_prime']     = wave[e_prime_idx]   # negative value
        features['a_prime']     = wave[a_prime_idx]   # negative value
        features['e_a_ratio']   = abs(features['e_prime']) / (abs(features['a_prime']) + 1e-6)
    elif len(peaks_neg) == 1:
        features['e_prime']     = wave[peaks_neg[0]]
        features['a_prime']     = wave[peaks_neg[0]]
        features['e_a_ratio']   = 1.0
    else:
        features['e_prime']     = features['min']
        features['a_prime']     = features['min']
        features['e_a_ratio']   = 1.0

    # Waveform shape: skewness and kurtosis
    from scipy.stats import skew, kurtosis
    features['skewness']  = skew(wave)
    features['kurtosis']  = kurtosis(wave)

    # Energy in frequency bands (via FFT)
    fft_mag = np.abs(np.fft.rfft(wave))
    freqs   = np.fft.rfftfreq(len(wave))
    low_band  = (freqs < 0.05)
    high_band = (freqs >= 0.05)
    features['energy_low']  = np.sum(fft_mag[low_band]**2)
    features['energy_high'] = np.sum(fft_mag[high_band]**2)
    features['energy_ratio'] = features['energy_low'] / (features['energy_high'] + 1e-6)

    return features

print("\n=== Extracting waveform features ===")
feature_list = [extract_waveform_features(X_clean[i]) for i in range(len(X_clean))]
df_wf = pd.DataFrame(feature_list)
print(f"Extracted {len(df_wf.columns)} features per patient")
print(df_wf.describe().round(3))

# ── Compare key features by outcome ───────────────────────────────────────
print("\n=== Key waveform features by outcome ===")
df_wf['CTRCD'] = y_clean
key_feats = ['s_prime', 'e_prime', 'e_a_ratio', 'mean', 'auc_ratio', 'energy_ratio']
for feat in key_feats:
    g0 = df_wf.loc[df_wf['CTRCD']==0, feat]
    g1 = df_wf.loc[df_wf['CTRCD']==1, feat]
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(g0, g1, alternative='two-sided')
    print(f"  {feat:15s}: no-CTRCD {g0.mean():.3f} | CTRCD {g1.mean():.3f} | p={p:.3f}")

# ── Cross-validate waveform features → logistic regression ────────────────
print("\n=== CV: waveform features → logistic regression ===")
feat_cols = [c for c in df_wf.columns if c != 'CTRCD']
X_feats   = df_wf[feat_cols].values
y_feats   = df_wf['CTRCD'].values

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, test_idx in skf.split(X_feats, y_feats):
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_feats[train_idx])
    X_te   = scaler.transform(X_feats[test_idx])
    clf    = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_feats[train_idx])
    prob   = clf.predict_proba(X_te)[:, 1]
    if y_feats[test_idx].sum() > 0:
        scores.append(roc_auc_score(y_feats[test_idx], prob))

wf_auc = np.mean(scores)
wf_std = np.std(scores)
print(f"Waveform features AUC: {wf_auc:.3f} ± {wf_std:.3f}")
print(f"(Note: AUC not directly comparable to C-index but indicative)")

# ══════════════════════════════════════════════════════════════════════════
# PART 3: 1D CNN on raw waveform
# Only run this if torch is available.
# At n=270 with ~27 events this is exploratory — don't over-interpret.
# The value is in seeing WHAT the CNN attends to, not just its AUC.
# ══════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("\n=== PyTorch available — running 1D CNN ===")
except ImportError:
    TORCH_AVAILABLE = False
    print("\n=== PyTorch not installed — skipping CNN ===")
    print("To run the CNN section: pip install torch")

if TORCH_AVAILABLE:

    class TDI_CNN(nn.Module):
        """
        Small 1D CNN for TDI waveform classification.
        Architecture kept deliberately shallow given n=270.
        Three conv layers → global average pooling → fully connected.
        Global average pooling (vs flatten) reduces parameter count
        and implicitly creates an attention map over the waveform.
        """
        def __init__(self, input_len=1001, n_filters=16):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, n_filters, kernel_size=15, padding=7),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.MaxPool1d(4)               # 1001 → 250
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(n_filters, n_filters*2, kernel_size=9, padding=4),
                nn.BatchNorm1d(n_filters*2),
                nn.ReLU(),
                nn.MaxPool1d(4)               # 250 → 62
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2),
                nn.BatchNorm1d(n_filters*4),
                nn.ReLU(),
                # Global average pooling: mean across time dimension
                # Output: (batch, n_filters*4)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(n_filters*4, 1)     # binary output
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.mean(dim=-1)               # global average pool
            return self.classifier(x).squeeze()

        def get_activation_map(self, x):
            """
            Returns the conv3 activation map — shows which parts of the
            waveform the CNN is attending to. Equivalent to a simple
            class activation map (CAM).
            """
            with torch.no_grad():
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)             # shape: (batch, channels, time)
            return x.mean(dim=1)              # average across channels

    def train_cnn(X_train, y_train, X_val, y_val,
                  epochs=50, lr=1e-3, batch_size=32):
        """Train one fold of the CNN, return val AUC and trained model."""
        # Scale to zero mean unit variance
        mean = X_train.mean()
        std  = X_train.std() + 1e-8
        X_tr = (X_train - mean) / std
        X_vl = (X_val   - mean) / std

        # Tensors: shape (N, 1, 1001) — 1 channel, 1001 timepoints
        Xt = torch.FloatTensor(X_tr).unsqueeze(1)
        yt = torch.FloatTensor(y_train)
        Xv = torch.FloatTensor(X_vl).unsqueeze(1)
        yv = torch.FloatTensor(y_val)

        # Class weights to handle 10% positive rate
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])

        model     = TDI_CNN()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        dataset = TensorDataset(Xt, yt)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_auc = 0
        best_state   = None

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            with torch.no_grad():
                logits = model(Xv)
                probs  = torch.sigmoid(logits).numpy()

            if yv.sum() > 0:
                val_auc = roc_auc_score(yv.numpy(), probs)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        return model, best_val_auc, (X_train.mean(), X_train.std() + 1e-8)

    # Cross-validate CNN
    print("Running 5-fold CV on CNN (this takes ~1-2 minutes)...")
    skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cnn_scores = []
    best_model = None
    best_auc   = 0
    best_scaler_params = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_clean, y_clean)):
        model, val_auc, scaler_params = train_cnn(
            X_clean[train_idx], y_clean[train_idx],
            X_clean[test_idx],  y_clean[test_idx],
            epochs=60
        )
        cnn_scores.append(val_auc)
        print(f"  Fold {fold+1}: AUC = {val_auc:.3f}")
        if val_auc > best_auc:
            best_auc          = val_auc
            best_model        = model
            best_scaler_params = scaler_params

    cnn_mean = np.mean(cnn_scores)
    cnn_std  = np.std(cnn_scores)
    print(f"\nCNN CV AUC: {cnn_mean:.3f} ± {cnn_std:.3f}")

    # ── Activation map: what does the CNN attend to? ────────────────────────
    if best_model is not None:
        print("\n=== Generating CNN activation maps ===")
        mean_scl, std_scl = best_scaler_params
        X_scaled = (X_clean - mean_scl) / std_scl
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)

        act_maps = best_model.get_activation_map(X_tensor).numpy()
        # act_maps shape: (N, time_after_conv2)
        # Upsample back to 1001 points for overlay with waveform
        from scipy.ndimage import zoom
        act_upsampled = np.array([
            zoom(act_maps[i], 1001 / act_maps.shape[1])
            for i in range(len(act_maps))
        ])

        mean_act_ctrcd    = act_upsampled[y_clean == 1].mean(axis=0)
        mean_act_no_ctrcd = act_upsampled[y_clean == 0].mean(axis=0)

        # Normalise to [0, 1] for visualisation
        def norm01(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor('#fafaf8')

        for ax, grp, label, wf_color, act_color in zip(
            axes,
            [0, 1],
            ['No CTRCD', 'CTRCD'],
            ['#1D9E75', '#D85A30'],
            ['#9FE1CB', '#F5C4B3']
        ):
            mask   = y_clean == grp
            mean_w = X_clean[mask].mean(axis=0)
            mean_a = norm01(act_upsampled[mask].mean(axis=0))

            ax2 = ax.twinx()
            ax2.fill_between(t, mean_a, alpha=0.35, color=act_color,
                             label='CNN attention')
            ax2.set_ylabel('Normalised attention', color=act_color, fontsize=9)
            ax2.set_ylim(0, 2.5)
            ax2.tick_params(axis='y', labelcolor=act_color)

            ax.plot(t, mean_w, color=wf_color, linewidth=2,
                    label=f'Mean waveform ({mask.sum()} patients)')
            ax.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
            ax.set_xlabel('Normalised cardiac cycle')
            ax.set_ylabel('Myocardial velocity (cm/s)')
            ax.set_title(f'{label} — waveform with CNN attention overlay',
                         fontweight='500')
            ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.savefig(OUT / 'tdi_cnn_attention.png', dpi=150,
                    bbox_inches='tight', facecolor='#fafaf8')
        print("Saved → results/tdi_cnn_attention.png")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n=== TDI model summary ===")
    print(f"  Waveform features (logistic):  AUC {wf_auc:.3f} ± {wf_std:.3f}")
    print(f"  1D CNN (raw waveform):         AUC {cnn_mean:.3f} ± {cnn_std:.3f}")
    print(f"  HFA-ICOS (clinical baseline):  AUC 0.663")
    print(f"\nNext step: fuse waveform features with tabular clinical")
    print(f"variables (script 05) for the combined model.")

else:
    # Without torch, just report waveform features result
    print("\n=== TDI model summary (no CNN) ===")
    print(f"  Waveform features (logistic):  AUC {wf_auc:.3f} ± {wf_std:.3f}")
    print(f"  HFA-ICOS (clinical baseline):  AUC 0.663")
    print(f"\nInstall torch to run the CNN: pip install torch")