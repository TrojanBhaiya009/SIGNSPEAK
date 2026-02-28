"""
train_combined.py
=================
Combined ASL fingerspelling + digit training pipeline.
Merges letter data (training_data/) with an external digit dataset (x.npy / y.npy)
and trains a single MLP that recognises A-Z AND 0-9.

Smart shape handling — auto-detects your npy format:
  • (N, 63)        → single-frame, one hand (ideal, used as-is)
  • (N, 87)        → enhanced features (stripped back to 63 + re-enhanced)
  • (N, 30, 126)   → collect_data.py LSTM sequences (middle frame extracted, right hand used)
  • (N, 30, 63)    → LSTM sequences, one hand (middle frame extracted)
  • (N, 126)       → flat both-hands (right hand 63:126 extracted)

Usage:
  python train_combined.py                              # auto-finds data
  python train_combined.py --digit-x x.npy --digit-y y.npy
  python train_combined.py --digit-x x.npy --digit-y y.npy --balance --augment
  python train_combined.py --digit-x x.npy --digit-y y.npy --all

  python train_combined.py --digit-x x.npy --digit-y y.npy --mirror       # both hands
  python train_combined.py --digit-x x.npy --digit-y y.npy --mirror --all  # full pipeline

Outputs:
  backend/sign_model.pkl               ← production model
  training_data/reports/               ← all evaluation charts
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.base import clone
from collections import Counter

# ───────────────────────────────────────────────────────────
#  PATHS  (edit if your layout differs)
# ───────────────────────────────────────────────────────────
LETTER_DATA_DIR = Path('training_data')      # X_train.npy + y_train.npy (letters)
MODEL_PATH      = Path('backend/sign_model.pkl')
REPORT_DIR      = Path('training_data/reports')

# Default digit npy locations (overridden by --digit-x / --digit-y)
DEFAULT_DIGIT_X = Path('x.npy')
DEFAULT_DIGIT_Y = Path('y.npy')

# Model hyper-params
HIDDEN_LAYERS = (256, 128, 64)
MAX_ITER      = 1500
EARLY_STOP    = True
VAL_FRACTION  = 0.12
NO_CHANGE     = 25

DEFAULT_THRESHOLD = 0.70

# ASL visually-similar pairs (letters + digits)
SIMILAR_PAIRS = [
    ('C', 'O'), ('U', 'V'), ('M', 'N'), ('S', 'A'),
    ('G', 'Q'), ('K', 'P'),
    ('1', 'L'), ('0', 'O'), ('2', 'V'), ('6', 'W'),
]


# ═══════════════════════════════════════════════════════════
#  SECTION 1 — Normalisation (must match sign_model.py)
# ═══════════════════════════════════════════════════════════

def normalize_landmarks(flat_63):
    """Wrist-centred + max-distance scale → position/scale invariant."""
    pts = np.array(flat_63, dtype=np.float64).reshape(21, 3)
    pts -= pts[0].copy()
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-6:
        pts /= max_dist
    return pts.flatten()


def compute_extra_features(normed_63):
    """24 engineered features (fingertip distances, curl, spread)."""
    pts  = normed_63.reshape(21, 3)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5,  9, 13, 17]
    feat = []
    # 10 pairwise fingertip distances
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            feat.append(np.linalg.norm(pts[tips[i]] - pts[tips[j]]))
    # 5 fingertip-to-wrist distances
    for t in tips:
        feat.append(np.linalg.norm(pts[t]))
    # 5 finger curl ratios (tip.y - mcp.y)
    for t, m in zip(tips, mcps):
        feat.append(pts[t, 1] - pts[m, 1])
    # 4 inter-finger spread angles
    for i in range(len(tips) - 1):
        v1 = pts[tips[i]] - pts[0]
        v2 = pts[tips[i + 1]] - pts[0]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        feat.append(cos_a)
    return np.array(feat, dtype=np.float64)


def build_87(raw_63):
    normed = normalize_landmarks(raw_63)
    extra  = compute_extra_features(normed)
    return np.concatenate([normed, extra])


def augment_sample(vec_63, n=4, noise_std=0.008, angle_range=0.12):
    """
    Simple data augmentation for a single 63-dim normalised vector.
    Returns n augmented copies:
      • Gaussian noise on joint positions
      • Small in-plane rotation around wrist
    """
    pts  = vec_63.reshape(21, 3)
    out  = []
    for _ in range(n):
        p = pts.copy()
        # Gaussian noise
        p += np.random.randn(*p.shape) * noise_std
        # In-plane (XY) rotation
        angle = np.random.uniform(-angle_range, angle_range)
        c, s  = np.cos(angle), np.sin(angle)
        R     = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p     = (R @ p.T).T
        # Re-normalise
        p -= p[0].copy()
        md = np.max(np.linalg.norm(p, axis=1))
        if md > 1e-6:
            p /= md
        out.append(p.flatten())
    return out


def mirror_sample(vec_63):
    """
    Mirror a right-hand landmark vector to simulate a left hand.
    Only X coordinate is flipped (x → -x), Y and Z stay the same.
    After flipping, re-normalise so scale invariance is preserved.
    This makes the model work for BOTH left-handed and right-handed users.
    """
    pts = vec_63.reshape(21, 3).copy()
    pts[:, 0] *= -1          # flip X axis only
    # Re-centre and re-normalise
    pts -= pts[0].copy()
    md = np.max(np.linalg.norm(pts, axis=1))
    if md > 1e-6:
        pts /= md
    return pts.flatten()


def add_mirror_augmentation(X, y):
    """
    Double the dataset by adding a mirrored copy of every sample.
    Original (right hand) + Mirror (left hand) = model works for both.
    """
    print("\n[MIRROR] Adding left-hand mirror augmentation...")
    print(f"  Before: {X.shape[0]} samples")
    X_mirror = np.array([mirror_sample(row) for row in X])
    X_out = np.vstack([X, X_mirror])
    y_out = np.concatenate([y, y])           # same labels for both hands
    # Shuffle so original and mirror are interleaved
    perm  = np.random.default_rng(77).permutation(len(X_out))
    print(f"  After : {X_out[perm].shape[0]} samples (2x — both hands covered)")
    return X_out[perm], y_out[perm]


# ═══════════════════════════════════════════════════════════
#  SECTION 2 — Data loading & shape normalisation
# ═══════════════════════════════════════════════════════════

def smart_load_npy(x_path, y_path, source_name="data"):
    """
    Load X and y npy files and return (X_63, y) where X_63 has shape (N, 63).
    Handles all known shape variants automatically.
    """
    x_path, y_path = Path(x_path), Path(y_path)
    if not x_path.exists():
        print(f"  [SKIP] {x_path} not found — skipping {source_name}")
        return None, None
    if not y_path.exists():
        print(f"  [SKIP] {y_path} not found — skipping {source_name}")
        return None, None

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True).astype(str)

    # Flatten object arrays (rare edge case from old numpy versions)
    if X.dtype == object:
        X = np.array(list(X), dtype=np.float32)

    shape = X.shape
    print(f"\n  [{source_name}] Raw shape: {shape}  |  Labels: {np.unique(y)}")

    # ── Handle different shapes ──────────────────────────────────────
    if X.ndim == 2 and shape[1] == 63:
        # Perfect: single-frame, one hand, already normalised or raw
        print(f"  [{source_name}] ✅ Shape (N,63) — using directly")
        X_out = X.astype(np.float64)

    elif X.ndim == 2 and shape[1] == 87:
        # Enhanced features — strip back to normalised 63
        print(f"  [{source_name}] ✅ Shape (N,87) — stripping extra 24 dims → 63")
        X_out = X[:, :63].astype(np.float64)

    elif X.ndim == 2 and shape[1] == 126:
        # Flat both-hands from collect_data.py (non-sequence version)
        print(f"  [{source_name}] ⚙️  Shape (N,126) — extracting right hand [63:126]")
        X_out = X[:, 63:].astype(np.float64)

    elif X.ndim == 3 and shape[2] == 63:
        # LSTM sequences, one hand: (N, frames, 63) → middle frame
        mid   = shape[1] // 2
        print(f"  [{source_name}] ⚙️  Shape (N,{shape[1]},63) — using middle frame [{mid}]")
        X_out = X[:, mid, :].astype(np.float64)

    elif X.ndim == 3 and shape[2] == 126:
        # LSTM sequences, both hands: (N, frames, 126) → middle frame, right hand
        mid   = shape[1] // 2
        print(f"  [{source_name}] ⚙️  Shape (N,{shape[1]},126) — middle frame [{mid}], right hand [63:126]")
        X_out = X[:, mid, 63:].astype(np.float64)

    else:
        print(f"  [{source_name}] ❌ Unrecognised shape {shape} — skipping")
        return None, None

    # Sanity-check: filter NaN/Inf rows
    valid = np.all(np.isfinite(X_out), axis=1)
    if not valid.all():
        n_bad = (~valid).sum()
        print(f"  [{source_name}] ⚠️  Removing {n_bad} NaN/Inf rows")
        X_out = X_out[valid]
        y     = y[valid]

    # Normalise each vector (idempotent if already normalised)
    X_norm = np.array([normalize_landmarks(row) for row in X_out])
    print(f"  [{source_name}] ✅ Final: {X_norm.shape}  labels: {dict(Counter(y))}")
    return X_norm, y


def load_letter_data():
    """Load existing letter training data (from train_alphabet.py convention)."""
    xp = LETTER_DATA_DIR / 'X_train.npy'
    yp = LETTER_DATA_DIR / 'y_train.npy'
    return smart_load_npy(xp, yp, "LETTERS")


def load_digit_data(x_path, y_path):
    """Load digit npy files."""
    return smart_load_npy(x_path, y_path, "DIGITS")


def merge_datasets(*pairs):
    """
    Merge any number of (X, y) pairs, skipping None entries.
    Returns (X_merged, y_merged) or raises if nothing to merge.
    """
    X_list, y_list = [], []
    for X, y in pairs:
        if X is not None and y is not None and len(X) > 0:
            X_list.append(X)
            y_list.append(y)
    if not X_list:
        print("\n[ERROR] No valid data found. Check file paths.")
        sys.exit(1)
    X_merged = np.vstack(X_list)
    y_merged = np.concatenate(y_list)
    return X_merged, y_merged


# ═══════════════════════════════════════════════════════════
#  SECTION 3 — Class balance
# ═══════════════════════════════════════════════════════════

def report_balance(y, title="Class distribution"):
    classes, counts = np.unique(y, return_counts=True)
    max_c    = counts.max()
    mean_c   = counts.mean()
    ratio    = max_c / counts.min()

    print(f"\n[BALANCE] {title}")
    print(f"  {'Label':<8} {'Count':>6}  Bar")

    # Sort: letters first, then digits, then other
    def sort_key(pair):
        c = pair[0]
        if c.isalpha():   return (0, c)
        if c.isdigit():   return (1, int(c))
        return (2, c)

    for cls, cnt in sorted(zip(classes, counts), key=sort_key):
        bar  = '█' * int(cnt / max_c * 28)
        flag = ' ⚠ LOW' if cnt < mean_c * 0.6 else ''
        print(f"  {cls:<8} {cnt:>6}  {bar}{flag}")

    print(f"\n  Min:{counts.min()}  Max:{counts.max()}  "
          f"Mean:{mean_c:.0f}  Ratio:{ratio:.1f}x")
    if ratio > 2.0:
        print("  ⚠ Imbalance ratio > 2 — strongly recommend --balance")
    elif ratio > 1.5:
        print("  ⚠ Mild imbalance — consider --balance")
    else:
        print("  ✅ Balance looks good")
    return classes, counts


def balance_classes(X, y, strategy='undersample'):
    """Undersample majority classes to the minority count."""
    classes, counts = np.unique(y, return_counts=True)
    min_c = counts.min()
    print(f"\n[BALANCE] Undersampling all classes to {min_c} samples")
    X_b, y_b = [], []
    for cls in classes:
        mask = y == cls
        Xi, yi = X[mask], y[mask]
        if len(Xi) > min_c:
            idx = resample(np.arange(len(Xi)), n_samples=min_c,
                           random_state=42, replace=False)
            Xi, yi = Xi[idx], yi[idx]
        X_b.append(Xi); y_b.append(yi)
    X_b = np.vstack(X_b)
    y_b = np.concatenate(y_b)
    perm = np.random.default_rng(42).permutation(len(X_b))
    return X_b[perm], y_b[perm]


def augment_dataset(X, y, n_aug=4):
    """
    Augment under-represented classes with noise+rotation variants.
    Over-represented classes are not augmented.
    """
    classes, counts = np.unique(y, return_counts=True)
    max_c = counts.max()
    print(f"\n[AUGMENT] Augmenting minority classes toward {max_c} samples")
    X_a, y_a = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt >= max_c:
            continue
        mask   = y == cls
        needed = max_c - cnt
        Xi     = X[mask]
        extras_X, extras_y = [], []
        while len(extras_X) < needed:
            for vec in Xi:
                aug = augment_sample(vec, n=1, noise_std=0.006, angle_range=0.10)
                extras_X.extend(aug)
                extras_y.append(cls)
                if len(extras_X) >= needed:
                    break
        X_a.append(np.array(extras_X[:needed]))
        y_a.append(np.array(extras_y[:needed]))
        print(f"  {cls}: {cnt} → {cnt + needed} (+{needed})")
    X_out = np.vstack(X_a)
    y_out = np.concatenate(y_a)
    perm  = np.random.default_rng(99).permutation(len(X_out))
    return X_out[perm], y_out[perm]


# ═══════════════════════════════════════════════════════════
#  SECTION 4 — Training
# ═══════════════════════════════════════════════════════════

def build_features(X_63, use_87=True):
    """Optionally expand 63-dim normalised to 87-dim enhanced."""
    if use_87:
        return np.array([build_87(row) for row in X_63])
    return X_63.copy()


def train_model(X_tr_sc, y_tr):
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOP,
        validation_fraction=VAL_FRACTION,
        n_iter_no_change=NO_CHANGE,
        random_state=42,
        verbose=False,
    )
    print(f"\n[TRAIN] Architecture  : {HIDDEN_LAYERS}")
    print(f"        Features      : {X_tr_sc.shape[1]}-dim")
    print(f"        Train samples : {len(X_tr_sc)}")
    model.fit(X_tr_sc, y_tr)
    print(f"        Stopped at    : iteration {model.n_iter_}")
    return model


def cross_validate(model, scaler, X, y_enc, classes, n_splits=5):
    """5-fold CV — returns (all_true, all_pred, mean_acc, std_acc)."""
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_t, all_p, accs = [], [], []
    print(f"\n[CV] {n_splits}-fold cross-validation…")
    for fold, (tri, tei) in enumerate(skf.split(X, y_enc), 1):
        sc = clone(scaler)
        Xtr = sc.fit_transform(X[tri])
        Xte = sc.transform(X[tei])
        m   = clone(model)
        m.fit(Xtr, y_enc[tri])
        pred = m.predict(Xte)
        acc  = (pred == y_enc[tei]).mean()
        accs.append(acc)
        print(f"  Fold {fold}: {acc:.1%}")
        all_t.extend(y_enc[tei])
        all_p.extend(pred)
    mean_a, std_a = np.mean(accs), np.std(accs)
    print(f"  CV Accuracy: {mean_a:.1%} ± {std_a:.1%}")
    return np.array(all_t), np.array(all_p), mean_a, std_a


# ═══════════════════════════════════════════════════════════
#  SECTION 5 — Threshold tuning
# ═══════════════════════════════════════════════════════════

def tune_threshold(model, X_te_sc, y_te_enc, le):
    print("\n[THRESHOLD] Evaluating confidence thresholds…")
    probas    = model.predict_proba(X_te_sc)
    max_prob  = probas.max(axis=1)
    pred_cls  = probas.argmax(axis=1)

    thresholds = np.arange(0.50, 0.96, 0.05)
    rows = []
    for t in thresholds:
        acc_mask = max_prob >= t
        n_acc    = acc_mask.sum()
        prec     = (pred_cls[acc_mask] == y_te_enc[acc_mask]).mean() if n_acc else 0.0
        cov      = n_acc / len(y_te_enc)
        rows.append((t, prec, cov, n_acc))

    print(f"\n  {'Thresh':>8}  {'Precision':>10}  {'Coverage':>10}  {'N accepted':>11}")
    print("  " + "-" * 46)
    for t, pr, co, n in rows:
        tag = " ◄ current" if abs(t - DEFAULT_THRESHOLD) < 0.025 else ""
        print(f"  {t:>8.2f}  {pr:>9.1%}  {co:>9.1%}  {n:>11}{tag}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ts   = [r[0] for r in rows]
    prec = [r[1] for r in rows]
    cov  = [r[2] for r in rows]
    ax.plot(ts, prec, 'o-',  color='#22c55e', label='Precision')
    ax.plot(ts, cov,  's--', color='#3b82f6', label='Coverage')
    ax.axvline(DEFAULT_THRESHOLD, color='#f59e0b', linestyle=':', lw=2,
               label=f'Current ({DEFAULT_THRESHOLD:.0%})')
    ax.set(xlabel='Confidence Threshold', ylabel='Rate',
           title='Precision vs Coverage by Threshold\n(A-Z + 0-9)',
           ylim=(0, 1.05), xticks=thresholds,
           xticklabels=[f'{t:.0%}' for t in thresholds])
    plt.xticks(rotation=30)
    for i, (t, p, c, _) in enumerate(rows):
        if p >= 0.85 and c >= 0.70:
            ax.axvspan(t - 0.025, t + 0.025, alpha=0.12, color='green')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = REPORT_DIR / 'threshold_tuning_combined.png'
    plt.savefig(out, dpi=130); plt.close()
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════
#  SECTION 6 — Visualisations
# ═══════════════════════════════════════════════════════════

def plot_class_distribution(y, classes, title, out_name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts  = [np.sum(y == c) for c in classes]
    mean_c  = np.mean(counts)
    colors  = ['#ef4444' if c < mean_c * 0.6 else '#22c55e' for c in counts]

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.55), 4),
                           facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.bar(classes, counts, color=colors, width=0.7)
    ax.axhline(mean_c, color='#f59e0b', linestyle='--', lw=1.5,
               label=f'Mean ({mean_c:.0f})')
    ax.tick_params(colors='#f1f5f9', labelsize=8)
    ax.set_title(title, color='#f1f5f9', fontsize=12, pad=10)
    ax.set_xlabel('Label', color='#94a3b8')
    ax.set_ylabel('Samples', color='#94a3b8')
    ax.legend(fontsize=9)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    plt.tight_layout()
    out = REPORT_DIR / out_name
    plt.savefig(out, dpi=130, facecolor='#0f172a'); plt.close()
    print(f"  Saved → {out}")


def plot_confusion_matrix(y_true, y_pred, classes, out_name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    n       = len(classes)
    cmap    = LinearSegmentedColormap.from_list('asl', ['#ffffff', '#4f46e5'])

    fig, ax = plt.subplots(figsize=(max(13, n * 0.5), max(11, n * 0.45)))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=max(6, 9 - n // 10))
    ax.set_yticklabels(classes, fontsize=max(6, 9 - n // 10))
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(f'Confusion Matrix — ASL A-Z + 0-9 (5-fold CV)\n'
                 f'{n} classes', fontsize=13, pad=12)

    # Cell annotations
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.02:
                ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                        fontsize=max(5, 7 - n // 12),
                        color='white' if v > thresh else '#1e1b4b',
                        fontweight='bold' if i == j else 'normal')

    # Highlight similar pairs
    for a, b in SIMILAR_PAIRS:
        if a in classes and b in classes:
            ai, bi = classes.index(a), classes.index(b)
            for r, c in [(ai, bi), (bi, ai)]:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     fill=False, edgecolor='#f59e0b',
                                     lw=2, linestyle='--')
                ax.add_patch(rect)

    plt.tight_layout()
    out = REPORT_DIR / out_name
    plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
    print(f"  Saved → {out}")


def plot_per_class_accuracy(class_acc, classes, mean_acc, std_acc, out_name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n      = len(classes)
    accs   = [class_acc.get(c, 0.0) for c in classes]
    colors = ['#ef4444' if a < 0.80 else '#f59e0b' if a < 0.92 else '#22c55e'
              for a in accs]

    # Separate letter and digit groups
    letters = [c for c in classes if c.isalpha()]
    digits  = [c for c in classes if c.isdigit()]

    fig, axes = plt.subplots(1, 2 if digits else 1,
                              figsize=(16, 5), facecolor='#0f172a')
    if not digits:
        axes = [axes]

    for ax, group, title in zip(
        axes,
        [letters, digits] if digits else [letters],
        ['Letters (A-Z)', 'Digits (0-9)'] if digits else ['Letters (A-Z)']
    ):
        ax.set_facecolor('#1e293b')
        grp_accs   = [class_acc.get(c, 0.0) for c in group]
        grp_colors = ['#ef4444' if a < 0.80 else '#f59e0b' if a < 0.92 else '#22c55e'
                      for a in grp_accs]
        bars = ax.bar(group, grp_accs, color=grp_colors, width=0.65)
        ax.axhline(0.92, color='#f59e0b', lw=1.2, linestyle='--', alpha=0.8)
        ax.axhline(0.80, color='#ef4444', lw=1.2, linestyle='--', alpha=0.8)
        ax.set_ylim(0, 1.12)
        ax.tick_params(colors='#f1f5f9')
        ax.set_title(f'{title}', color='#f1f5f9', fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.set_ylabel('Accuracy', color='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        for bar, acc in zip(bars, grp_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f'{acc:.0%}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#f1f5f9')

    fig.suptitle(f'Per-Class Accuracy — A-Z + 0-9 | '
                 f'CV: {mean_acc:.1%} ± {std_acc:.1%}',
                 color='#f1f5f9', fontsize=13, y=1.01)
    plt.tight_layout()
    out = REPORT_DIR / out_name
    plt.savefig(out, dpi=140, facecolor='#0f172a', bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


def plot_dashboard(y_true, y_pred, classes, class_acc, mean_acc, std_acc,
                   n_total, out_name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n       = len(classes)
    letters = [c for c in classes if c.isalpha()]
    digits  = [c for c in classes if c.isdigit()]

    fig  = plt.figure(figsize=(20, 11), facecolor='#0f172a')
    TEXT = '#f1f5f9'; CARD = '#1e293b'; GREEN = '#22c55e'
    AMBER = '#f59e0b'; RED = '#ef4444'; INDIGO = '#6366f1'

    fig.text(0.5, 0.96,
             'ASL Fingerspelling — Combined Letters + Digits | Model Evaluation',
             ha='center', fontsize=17, fontweight='bold',
             color=TEXT, family='monospace')

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           left=0.05, right=0.97, top=0.91, bottom=0.07,
                           wspace=0.35, hspace=0.50)

    # ── KPI row ──
    kpis = [
        ('CV Accuracy', f'{mean_acc:.1%}', f'± {std_acc:.1%}',  GREEN),
        ('Total Classes', str(n),          f'{len(letters)}L + {len(digits)}D', INDIGO),
        ('Train Samples', f'{n_total:,}',  'across 5 folds', AMBER),
        ('Best Class', max(class_acc, key=class_acc.get),
         f'{max(class_acc.values()):.0%}', GREEN),
    ]
    for col, (lbl, big, small, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(CARD); ax.axis('off')
        ax.text(0.5, 0.70, big,   ha='center', fontsize=26, fontweight='bold',
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.40, lbl,   ha='center', fontsize=10, color=TEXT,
                transform=ax.transAxes)
        ax.text(0.5, 0.16, small, ha='center', fontsize=8,  color='#94a3b8',
                transform=ax.transAxes)

    # ── Accuracy bar (bottom, span all) ──
    ax2 = fig.add_subplot(gs[1, :3])
    ax2.set_facecolor(CARD)
    accs   = [class_acc.get(c, 0.0) for c in classes]
    colors = [RED if a < 0.80 else AMBER if a < 0.92 else GREEN for a in accs]
    ax2.bar(classes, accs, color=colors, width=0.7)
    ax2.axhline(0.92, color=AMBER, lw=1, linestyle='--', alpha=0.7, label='92%')
    ax2.axhline(0.80, color=RED,   lw=1, linestyle='--', alpha=0.7, label='80%')
    ax2.set_ylim(0, 1.14)
    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.set_title('Per-Label Accuracy  (green ≥92%, amber ≥80%, red <80%)',
                  color=TEXT, fontsize=11, pad=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax2.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD)
    for sp in ax2.spines.values(): sp.set_color('#334155')

    # ── Confused pairs table ──
    ax3 = fig.add_subplot(gs[1, 3])
    ax3.set_facecolor(CARD); ax3.axis('off')
    ax3.set_title('Similar-Sign Pairs', color=TEXT, fontsize=11, pad=8)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    rows = []
    for a, b in SIMILAR_PAIRS:
        if a in classes and b in classes:
            ai, bi = classes.index(a), classes.index(b)
            err = max(cm_norm[ai, bi], cm_norm[bi, ai])
            rows.append((a, b, err))
    rows.sort(key=lambda r: -r[2])
    for i, (a, b, err) in enumerate(rows[:8]):
        yp = 0.88 - i * 0.11
        color = RED if err > 0.10 else AMBER if err > 0.03 else GREEN
        ax3.text(0.05, yp, f'{a}  ↔  {b}', fontsize=12, fontweight='bold',
                 color=TEXT, transform=ax3.transAxes, va='center')
        ax3.text(0.58, yp, f'{err:.0%}', fontsize=10, color=color,
                 transform=ax3.transAxes, va='center')

    plt.savefig(REPORT_DIR / out_name, dpi=150, bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"  Saved → {REPORT_DIR / out_name}")


# ═══════════════════════════════════════════════════════════
#  SECTION 7 — Learning curves
# ═══════════════════════════════════════════════════════════

def plot_learning_curves(model, X_sc, y_enc, out_name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f172a')
    for ax in axes: ax.set_facecolor('#1e293b')

    # Loss curve
    ax = axes[0]
    ax.plot(model.loss_curve_, color='#3b82f6', label='Train loss')
    if model.validation_scores_:
        val_loss = 1 - np.array(model.validation_scores_)
        ax.plot(val_loss, color='#ef4444', linestyle='--', label='Val loss proxy')
        gap = val_loss[-1] - model.loss_curve_[-1]
        color = '#ef4444' if gap > 0.1 else '#22c55e'
        ax.annotate(f'Gap {gap:.3f}',
                    xy=(len(model.loss_curve_) - 1, val_loss[-1]),
                    xytext=(-60, 15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color),
                    color=color, fontweight='bold')
    ax.set_title('Loss Curve', color='#f1f5f9')
    ax.tick_params(colors='#f1f5f9')
    ax.legend(labelcolor='#f1f5f9', facecolor='#1e293b')
    ax.grid(alpha=0.2)

    # sklearn learning curve
    ax = axes[1]
    fast_m = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS,
                            max_iter=400, early_stopping=True,
                            validation_fraction=VAL_FRACTION, random_state=42)
    sizes, tr_sc, va_sc = learning_curve(
        fast_m, X_sc, y_enc, cv=3,
        train_sizes=np.linspace(0.2, 1.0, 6),
        scoring='accuracy', n_jobs=-1)
    tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
    va_m, va_s = va_sc.mean(1), va_sc.std(1)
    ax.plot(sizes, tr_m, 'o-', color='#3b82f6', label='Train acc')
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.12, color='#3b82f6')
    ax.plot(sizes, va_m, 's--', color='#ef4444', label='Val acc')
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.12, color='#ef4444')
    ax.set_ylim(0, 1.05)
    ax.set_title('Learning Curve', color='#f1f5f9')
    ax.tick_params(colors='#f1f5f9')
    ax.legend(labelcolor='#f1f5f9', facecolor='#1e293b')
    ax.grid(alpha=0.2)

    gap = tr_m[-1] - va_m[-1]
    msg = (f'⚠ Overfitting (gap={gap:.2f})' if gap > 0.15
           else f'⚠ Low val acc ({va_m[-1]:.0%})' if va_m[-1] < 0.75
           else f'✓ Model healthy (val={va_m[-1]:.0%}, gap={gap:.2f})')
    fig.text(0.5, 0.01, msg, ha='center', fontsize=10,
             color='#ef4444' if gap > 0.15 else '#22c55e')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = REPORT_DIR / out_name
    plt.savefig(out, dpi=130, facecolor='#0f172a'); plt.close()
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Combined ASL Letters + Digits Training Pipeline')
    parser.add_argument('--digit-x',  default=str(DEFAULT_DIGIT_X),
                        help='Path to digit X.npy  (default: x.npy)')
    parser.add_argument('--digit-y',  default=str(DEFAULT_DIGIT_Y),
                        help='Path to digit y.npy  (default: y.npy)')
    parser.add_argument('--letter-x', default=str(LETTER_DATA_DIR / 'X_train.npy'),
                        help='Path to letter X_train.npy')
    parser.add_argument('--letter-y', default=str(LETTER_DATA_DIR / 'y_train.npy'),
                        help='Path to letter y_train.npy')
    parser.add_argument('--mirror', action='store_true',
                        help='Add mirrored (left-hand) copy of all samples — model works for both hands')
    parser.add_argument('--no-87',  action='store_true',
                        help='Use 63-dim features only (skip extra 24 features)')
    parser.add_argument('--balance',  action='store_true',
                        help='Undersample to balance class counts')
    parser.add_argument('--augment',  action='store_true',
                        help='Augment minority classes with noise+rotation')
    parser.add_argument('--threshold-tune', action='store_true',
                        help='Tune confidence threshold and save chart')
    parser.add_argument('--no-cv',   action='store_true',
                        help='Skip 5-fold CV (faster, less honest eval)')
    parser.add_argument('--all',     action='store_true',
                        help='Enable all optional steps')
    args = parser.parse_args()

    use_87 = not args.no_87
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  COMBINED TRAINING: ASL A-Z + 0-9")
    print("=" * 65)

    # ── 1. Load ──────────────────────────────────────────────────────
    print("\n[LOAD] Loading datasets…")
    X_let, y_let = load_letter_data()
    X_dig, y_dig = load_digit_data(args.digit_x, args.digit_y)

    if X_let is None and X_dig is None:
        print("\n[ERROR] No data found at all. Provide at least one dataset.")
        sys.exit(1)

    X, y = merge_datasets((X_let, y_let), (X_dig, y_dig))
    print(f"\n[MERGE] Combined dataset: {X.shape}  |  "
          f"Classes: {sorted(np.unique(y))}")

    # ── 2. Balance / augment ─────────────────────────────────────────
    classes_sorted = sorted(np.unique(y),
                            key=lambda c: (0, c) if c.isalpha() else (1, int(c)))
    report_balance(y, "Before preprocessing")
    plot_class_distribution(y, classes_sorted,
                            'Class Distribution (before)', 'dist_before.png')

    if args.balance or args.all:
        X, y = balance_classes(X, y)
    if args.augment or args.all:
        X, y = augment_dataset(X, y)

    report_balance(y, "After preprocessing")
    plot_class_distribution(y, sorted(np.unique(y),
                            key=lambda c: (0,c) if c.isalpha() else (1,int(c))),
                            'Class Distribution (after)', 'dist_after.png')

    # ── 2b. Mirror augmentation (left-hand support) ─────────────────────
    if args.mirror or args.all:
        X, y = add_mirror_augmentation(X, y)

    # ── 3. Feature engineering ───────────────────────────────────────
    print(f"\n[FEATURES] Building {'87' if use_87 else '63'}-dim features…")
    X_feat = build_features(X, use_87=use_87)
    print(f"  Feature matrix: {X_feat.shape}")

    # ── 4. Encode & split ────────────────────────────────────────────
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y_enc, test_size=0.20, random_state=42, stratify=y_enc)

    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_tr)
    X_te_sc    = scaler.transform(X_te)

    # ── 5. Train ──────────────────────────────────────────────────────
    model = train_model(X_tr_sc, y_tr)

    tr_acc = model.score(X_tr_sc, y_tr)
    te_acc = model.score(X_te_sc, y_te)
    gap    = tr_acc - te_acc
    print(f"\n[RESULTS]")
    print(f"  Train acc : {tr_acc:.1%}")
    print(f"  Test  acc : {te_acc:.1%}")
    gap_msg = ('⚠ Overfitting' if gap > 0.15
               else '⚠ Mild overfit' if gap > 0.07 else '✅ Healthy')
    print(f"  Gap       : {gap:.1%}  {gap_msg}")

    print("\n[REPORT] Per-class (hold-out test set):")
    y_pred_te = model.predict(X_te_sc)
    print(classification_report(y_te, y_pred_te, target_names=classes))

    # ── 6. Cross-validation ───────────────────────────────────────────
    if not args.no_cv:
        y_true_cv, y_pred_cv, mean_acc, std_acc = cross_validate(
            model, scaler, X_feat, y_enc, classes, n_splits=5)
    else:
        y_true_cv  = y_te
        y_pred_cv  = y_pred_te
        mean_acc   = te_acc
        std_acc    = 0.0

    # Per-class accuracy (from CV)
    class_acc = {}
    for i, cls in enumerate(classes):
        mask = y_true_cv == i
        class_acc[cls] = float((y_pred_cv[mask] == i).mean()) if mask.any() else 0.0

    # ── 7. Threshold tuning ───────────────────────────────────────────
    if args.threshold_tune or args.all:
        tune_threshold(model, X_te_sc, y_te, le)

    # ── 8. Plots ──────────────────────────────────────────────────────
    print("\n[PLOTS] Generating charts…")
    plot_learning_curves(model, X_tr_sc, y_tr, 'learning_curves_combined.png')
    plot_confusion_matrix(y_true_cv, y_pred_cv, classes,
                          'confusion_matrix_combined.png')
    plot_per_class_accuracy(class_acc, classes, mean_acc, std_acc,
                            'per_class_accuracy_combined.png')
    plot_dashboard(y_true_cv, y_pred_cv, classes, class_acc,
                   mean_acc, std_acc, len(X_feat),
                   'dashboard_combined.png')

    # ── 9. Save model ─────────────────────────────────────────────────
    model_data = {
        'model':          model,
        'scaler':         scaler,
        'classes':        classes,
        'label_encoder':  le,
        'test_accuracy':  te_acc,
        'cv_accuracy':    mean_acc,
        'cv_std':         std_acc,
        'feature_dim':    X_feat.shape[1],
        'uses_87':        use_87,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n[SAVE] Model  → {MODEL_PATH}")
    print(f"[SAVE] Reports→ {REPORT_DIR}/")
    print(f"\n{'=' * 65}")
    print(f"  ✅ DONE!")
    print(f"  Classes : {len(classes)}  ({sum(c.isalpha() for c in classes)} letters"
          f" + {sum(c.isdigit() for c in classes)} digits)")
    print(f"  CV acc  : {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  Feature : {X_feat.shape[1]}-dim")
    print(f"  Model   : {MODEL_PATH}")
    print(f"{'=' * 65}\n")
    print("  👉 Open training_data/reports/dashboard_combined.png for a summary")


if __name__ == '__main__':
    main()