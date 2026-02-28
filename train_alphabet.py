"""
train_alphabet.py
=================
Standalone training pipeline for ASL fingerspelling recognition.
Loads data collected by collect_data.py and trains/evaluates the model.

Addresses:
  - Class imbalance detection & balancing
  - Overfitting monitoring (train vs val accuracy curves)
  - Confidence threshold tuning (ROC / precision-recall per class)

Usage:
  python train_alphabet.py                  # train with defaults
  python train_alphabet.py --balance        # undersample to balance classes
  python train_alphabet.py --threshold-tune # find optimal confidence threshold
  python train_alphabet.py --all            # run everything
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
from sklearn.utils import resample

# ──────────────────────────────────────────────
# CONFIG — edit these to tune the pipeline
# ──────────────────────────────────────────────
DATA_DIR      = Path('training_data')
MODEL_PATH    = Path('backend/sign_model.pkl')
REPORT_DIR    = Path('training_data/reports')

# Model architecture
HIDDEN_LAYERS = (256, 128, 64)   # reduce to (128, 64) if overfitting
MAX_ITER      = 1000
EARLY_STOP    = True
VAL_FRACTION  = 0.15
NO_CHANGE     = 20

# Threshold tuning
DEFAULT_THRESHOLD = 0.70         # must match app.py
# ──────────────────────────────────────────────


def load_data():
    """Load X_train.npy / y_train.npy saved by collect_data.py"""
    x_path = DATA_DIR / 'X_train.npy'
    y_path = DATA_DIR / 'y_train.npy'

    if not x_path.exists() or not y_path.exists():
        print(f"[ERROR] No training data found in '{DATA_DIR}'.")
        print("        Run collect_data.py first and press S to save.")
        sys.exit(1)

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"[DATA] Loaded {len(X)} samples, {X.shape[1]} features each.")
    return X, y


# ──────────────────────────────────────────────
# 1. CLASS IMBALANCE
# ──────────────────────────────────────────────

def check_class_balance(y, plot=True):
    """Print and optionally plot class distribution."""
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()
    mean_count = counts.mean()
    imbalance_ratio = max_count / min_count

    print("\n[BALANCE] Class distribution:")
    print(f"  {'Letter':<8} {'Count':>6}  {'Bar'}")
    for cls, cnt in zip(classes, counts):
        bar = '█' * int(cnt / max_count * 30)
        flag = ' ⚠ LOW' if cnt < mean_count * 0.7 else ''
        print(f"  {cls:<8} {cnt:>6}  {bar}{flag}")
    print(f"\n  Min: {min_count}  Max: {max_count}  "
          f"Ratio: {imbalance_ratio:.2f}x  Mean: {mean_count:.0f}")

    if imbalance_ratio > 1.5:
        print("\n  [WARNING] Imbalance ratio > 1.5 — consider using --balance")
    else:
        print("\n  [OK] Class balance looks good.")

    if plot:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 4))
        colors = ['#e74c3c' if c < mean_count * 0.7 else '#2ecc71' for c in counts]
        ax.bar(classes, counts, color=colors)
        ax.axhline(mean_count, color='orange', linestyle='--', label=f'Mean ({mean_count:.0f})')
        ax.axhline(mean_count * 0.7, color='red', linestyle=':', label='70% of mean (warning)')
        ax.set_title('Class Distribution')
        ax.set_xlabel('Letter')
        ax.set_ylabel('Sample Count')
        ax.legend()
        plt.tight_layout()
        out = REPORT_DIR / 'class_distribution.png'
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Saved → {out}")

    return classes, counts


def balance_classes(X, y):
    """Undersample majority classes to match the minority class count."""
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    print(f"\n[BALANCE] Undersampling all classes to {min_count} samples each.")

    X_bal, y_bal = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        idx_sampled = resample(idx, n_samples=min_count, random_state=42, replace=False)
        X_bal.append(X[idx_sampled])
        y_bal.append(y[idx_sampled])

    X_bal = np.vstack(X_bal)
    y_bal = np.concatenate(y_bal)
    # Shuffle
    perm = np.random.default_rng(42).permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# ──────────────────────────────────────────────
# 2. TRAINING + OVERFITTING MONITORING
# ──────────────────────────────────────────────

def train_model(X_train_sc, y_train):
    """Train MLPClassifier and return the fitted model."""
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOP,
        validation_fraction=VAL_FRACTION,
        n_iter_no_change=NO_CHANGE,
        random_state=42,
        verbose=False,   # we'll print our own summary
    )
    print(f"\n[TRAIN] Architecture: {HIDDEN_LAYERS}")
    print(f"        Max iterations: {MAX_ITER}, early stopping: {EARLY_STOP}")
    model.fit(X_train_sc, y_train)
    print(f"        Stopped at iteration: {model.n_iter_}")
    return model


def plot_learning_curves(model, X_train_sc, y_train):
    """
    Plot train vs validation loss (from MLPClassifier loss curves)
    and sklearn learning curves (accuracy vs training size).
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # — Loss curve (from MLPClassifier internals) —
    ax = axes[0]
    ax.plot(model.loss_curve_, label='Train loss', color='#3498db')
    if model.validation_scores_ is not None:
        # validation_scores_ is accuracy; invert to get "val loss" proxy
        val_loss_proxy = 1 - np.array(model.validation_scores_)
        ax.plot(val_loss_proxy, label='Val loss (1-acc)', color='#e74c3c', linestyle='--')
    ax.set_title('Loss Curve')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Gap analysis annotation
    if model.validation_scores_ is not None:
        final_train_loss = model.loss_curve_[-1]
        final_val_loss   = 1 - model.validation_scores_[-1]
        gap = final_val_loss - final_train_loss
        color = '#e74c3c' if gap > 0.1 else '#2ecc71'
        ax.annotate(f'Gap: {gap:.3f}',
                    xy=(len(model.loss_curve_) - 1, final_val_loss),
                    xytext=(-80, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color),
                    color=color, fontweight='bold')

    # — sklearn learning curve (accuracy vs sample size) —
    ax = axes[1]
    # Use a fast n_jobs=-1; limit max_iter to keep it quick
    fast_model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        max_iter=300,
        early_stopping=True,
        validation_fraction=VAL_FRACTION,
        random_state=42,
    )
    sizes, train_scores, val_scores = learning_curve(
        fast_model, X_train_sc, y_train,
        cv=3, train_sizes=np.linspace(0.2, 1.0, 6),
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(sizes, train_mean, 'o-', color='#3498db', label='Train accuracy')
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#3498db')
    ax.plot(sizes, val_mean, 's--', color='#e74c3c', label='Val accuracy')
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#e74c3c')
    ax.set_title('Learning Curve (Accuracy vs Data Size)')
    ax.set_xlabel('Training samples')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Overfitting advice
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.15:
        fig.text(0.5, 0.01,
                 f'⚠  Overfitting detected (gap={final_gap:.2f}). '
                 'Try smaller network, more data, or stronger regularization.',
                 ha='center', color='red', fontsize=10)
    elif val_mean[-1] < 0.75:
        fig.text(0.5, 0.01,
                 f'⚠  Low val accuracy ({val_mean[-1]:.0%}). Collect more/better samples.',
                 ha='center', color='orange', fontsize=10)
    else:
        fig.text(0.5, 0.01,
                 f'✓  Model looks healthy (val acc={val_mean[-1]:.0%}, gap={final_gap:.2f}).',
                 ha='center', color='green', fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = REPORT_DIR / 'learning_curves.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out}")


def plot_confusion_matrix(y_true, y_pred, classes):
    """Heatmap confusion matrix with per-class accuracy on diagonal."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix (counts)', 'Confusion Matrix (normalised)'],
        ['d', '.2f']
    ):
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        thresh = data.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = data[i, j]
                text = format(val, fmt) if fmt == '.2f' else str(val)
                ax.text(j, i, text, ha='center', va='center', fontsize=6,
                        color='white' if val > thresh else 'black')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = REPORT_DIR / 'confusion_matrix.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out}")

    # Print worst confusions
    np.fill_diagonal(cm_norm, 0)
    worst_idx = np.dstack(np.unravel_index(np.argsort(cm_norm.ravel())[::-1], cm_norm.shape))[0][:5]
    print("\n  Top-5 most confused pairs (True → Predicted):")
    for true_i, pred_i in worst_idx:
        if cm_norm[true_i, pred_i] > 0:
            print(f"    {classes[true_i]} → {classes[pred_i]}: "
                  f"{cm_norm[true_i, pred_i]:.0%} of true {classes[true_i]} samples")


# ──────────────────────────────────────────────
# 3. CONFIDENCE THRESHOLD TUNING
# ──────────────────────────────────────────────

def tune_threshold(model, X_test_sc, y_test, classes):
    """
    For each threshold value, calculate:
      - Recall (fraction of signs that get *any* prediction)
      - Precision (fraction of predictions that are correct)
    and print a table to help choose the right threshold for app.py.
    """
    print("\n[THRESHOLD] Evaluating confidence thresholds...")
    probas = model.predict_proba(X_test_sc)
    max_prob = probas.max(axis=1)
    pred_class = probas.argmax(axis=1)

    thresholds = np.arange(0.50, 0.96, 0.05)
    rows = []
    for t in thresholds:
        accepted = max_prob >= t
        n_accepted = accepted.sum()
        if n_accepted == 0:
            precision = 0.0
        else:
            precision = (pred_class[accepted] == y_test[accepted]).mean()
        recall = n_accepted / len(y_test)   # fraction of signs not discarded
        rows.append((t, precision, recall, n_accepted))

    print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Coverage':>10}  {'Accepted':>9}")
    print("  " + "-" * 46)
    for t, prec, rec, n in rows:
        marker = " ◄ current" if abs(t - DEFAULT_THRESHOLD) < 0.025 else ""
        flag   = " ⚠ low coverage" if rec < 0.6 else (" ⚠ low precision" if prec < 0.80 else "")
        print(f"  {t:>10.2f}  {prec:>9.1%}  {rec:>9.1%}  {n:>9}{marker}{flag}")

    # Plot precision-coverage tradeoff
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ts   = [r[0] for r in rows]
    prec = [r[1] for r in rows]
    cov  = [r[2] for r in rows]
    ax.plot(ts, prec, 'o-', color='#2ecc71',  label='Precision (correct when shown)')
    ax.plot(ts, cov,  's--', color='#3498db', label='Coverage (fraction of signs shown)')
    ax.axvline(DEFAULT_THRESHOLD, color='orange', linestyle=':', linewidth=2,
               label=f'Current threshold ({DEFAULT_THRESHOLD:.0%})')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('Precision vs Coverage by Confidence Threshold\n'
                 '(choose where both curves are acceptably high)')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.0%}' for t in thresholds], rotation=30)
    ax.legend()
    ax.grid(alpha=0.3)

    # Shade "sweet spot" — precision ≥ 0.85 AND coverage ≥ 0.70
    for i, (t, p, c, _) in enumerate(rows):
        if p >= 0.85 and c >= 0.70:
            ax.axvspan(t - 0.025, t + 0.025, alpha=0.15, color='green')

    plt.tight_layout()
    out = REPORT_DIR / 'threshold_tuning.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n  Saved → {out}")
    print("  Green shaded region = thresholds where precision ≥ 85% AND coverage ≥ 70%")
    print(f"  To update: change CONFIDENCE_THRESHOLD in app.py from {DEFAULT_THRESHOLD:.0%}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ASL Fingerspelling Training Pipeline')
    parser.add_argument('--balance',        action='store_true', help='Undersample to balance classes')
    parser.add_argument('--threshold-tune', action='store_true', help='Tune confidence threshold')
    parser.add_argument('--all',            action='store_true', help='Run all steps')
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    X, y = load_data()

    # ── 1. Class balance ──
    check_class_balance(y, plot=True)

    if args.balance or args.all:
        X, y = balance_classes(X, y)
        check_class_balance(y, plot=False)

    # ── Encode + split ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 2. Train + overfitting check ──
    model = train_model(X_train_sc, y_train)

    train_acc = model.score(X_train_sc, y_train)
    test_acc  = model.score(X_test_sc,  y_test)
    gap       = train_acc - test_acc

    print(f"\n[RESULTS]")
    print(f"  Train accuracy : {train_acc:.1%}")
    print(f"  Test  accuracy : {test_acc:.1%}")
    print(f"  Overfit gap    : {gap:.1%}", end='')
    if gap > 0.15:
        print("  ⚠  Overfitting — try smaller network or more data")
    elif gap > 0.07:
        print("  ⚠  Mild overfitting — monitor")
    else:
        print("  ✓  Healthy")

    print("\n[REPORT] Per-class metrics:")
    y_pred = model.predict(X_test_sc)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Plots ──
    print("[PLOTS] Generating visualisations...")
    plot_learning_curves(model, X_train_sc, y_train)
    plot_confusion_matrix(y_test, y_pred, le.classes_)

    # ── 3. Threshold tuning ──
    if args.threshold_tune or args.all:
        tune_threshold(model, X_test_sc, y_test, le.classes_)

    # ── Save model ──
    model_data = {
        'model':         model,
        'scaler':        scaler,
        'classes':       list(le.classes_),
        'label_encoder': le,
        'test_accuracy': test_acc,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n[SAVE] Model saved → {MODEL_PATH}")
    print(f"[SAVE] Reports     → {REPORT_DIR}/")
    print("\nDone! Open training_data/reports/ to review all charts.")


if __name__ == '__main__':
    main()