"""
evaluate_model.py
=================
Presentation-ready evaluation of the trained ASL fingerspelling model.

Generates:
  1. Per-class accuracy table (terminal)
  2. Confusion matrix heatmap  (training_data/reports/eval_confusion_matrix.png)
  3. Per-class accuracy bar chart (training_data/reports/eval_per_class_accuracy.png)
  4. Summary dashboard  (training_data/reports/eval_dashboard.png)  ← show this one

Usage:
  python evaluate_model.py
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path('training_data')
MODEL_PATH = Path('backend/sign_model.pkl')
REPORT_DIR = Path('training_data/reports')

# ── ASL pairs that are visually similar (highlighted in output) ────────────
SIMILAR_PAIRS = [('C', 'O'), ('U', 'V'), ('M', 'N'), ('S', 'A'), ('G', 'Q'), ('K', 'P')]


def load_artifacts():
    """Load model + raw training data."""
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}. Run train_alphabet.py first.")
        sys.exit(1)
    if not (DATA_DIR / 'X_train.npy').exists():
        print(f"[ERROR] Training data not found in {DATA_DIR}. Run collect_data.py first.")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)

    model   = data['model']
    scaler  = data['scaler']
    classes = data['classes']

    X = np.load(DATA_DIR / 'X_train.npy')
    y = np.load(DATA_DIR / 'y_train.npy')

    n_feat = X.shape[1] if X.ndim > 1 else 0
    print(f"[EVAL] Loaded {len(X)} samples | {n_feat}-dim features | "
          f"{len(classes)} classes | model: sklearn MLP")
    return model, scaler, classes, X, y


def cross_val_evaluate(model, scaler, classes, X, y, n_splits=5):
    """
    5-fold cross-validation for honest per-class accuracy.
    Returns y_true and y_pred arrays across all folds.
    """
    from sklearn.base import clone
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Encode labels
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([class_to_idx[label] for label in y])

    all_true, all_pred = [], []
    fold_accs = []

    print(f"\n[EVAL] Running {n_splits}-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        sc = clone(scaler)
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        m = clone(model)
        m.fit(X_tr_sc, y_tr)

        preds = m.predict(X_te_sc)
        acc   = (preds == y_te).mean()
        fold_accs.append(acc)
        print(f"  Fold {fold}: {acc:.1%}")

        all_true.extend(y_te)
        all_pred.extend(preds)

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  CV Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    return np.array(all_true), np.array(all_pred), mean_acc, std_acc


def per_class_accuracy(y_true, y_pred, classes):
    """Return dict of letter → accuracy."""
    results = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() == 0:
            results[cls] = 0.0
        else:
            results[cls] = (y_pred[mask] == i).mean()
    return results


def print_per_class_table(class_acc, classes):
    """Print a tidy terminal table with similar-pair warnings."""
    similar_letters = {l for pair in SIMILAR_PAIRS for l in pair}

    print("\n" + "=" * 52)
    print("  PER-CLASS ACCURACY (5-fold CV)")
    print("=" * 52)
    print(f"  {'Letter':<8} {'Accuracy':>9}  {'Bar':<26}  Note")
    print("  " + "-" * 48)

    for cls in classes:
        acc  = class_acc[cls]
        bar  = '█' * int(acc * 25)
        note = ''
        if acc < 0.85:
            note = '⚠ LOW'
        elif cls in similar_letters:
            pairs = [p for p in SIMILAR_PAIRS if cls in p]
            other = [c for p in pairs for c in p if c != cls]
            note  = f'similar to {", ".join(other)}'
        color = '\033[91m' if acc < 0.85 else ('\033[93m' if acc < 0.95 else '\033[92m')
        reset = '\033[0m'
        print(f"  {cls:<8} {color}{acc:>8.1%}{reset}  {bar:<26}  {note}")

    print("=" * 52)


# ── Plot 1: Confusion Matrix ───────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Custom colormap: white → indigo
    cmap = LinearSegmentedColormap.from_list('asl', ['#ffffff', '#4f46e5'])

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel('Predicted Letter', fontsize=12)
    ax.set_ylabel('True Letter', fontsize=12)
    ax.set_title('Confusion Matrix — ASL Fingerspelling (5-fold CV)', fontsize=14, pad=15)

    thresh = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            val  = cm_norm[i, j]
            text = f'{val:.0%}' if val > 0.02 else ''
            ax.text(j, i, text, ha='center', va='center', fontsize=7,
                    color='white' if val > thresh else '#1e1b4b',
                    fontweight='bold' if i == j else 'normal')

    # Highlight similar pairs
    for a, b in SIMILAR_PAIRS:
        if a in classes and b in classes:
            ai, bi = classes.index(a), classes.index(b)
            for (row, col) in [(ai, bi), (bi, ai)]:
                rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                     fill=False, edgecolor='#f59e0b',
                                     linewidth=2.5, linestyle='--')
                ax.add_patch(rect)

    # Legend for highlights
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='none', edgecolor='#f59e0b',
                    linestyle='--', linewidth=2, label='Visually similar pair')]
    ax.legend(handles=legend, loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── Plot 2: Per-class accuracy bar chart ──────────────────────────────────

def plot_per_class_accuracy(class_acc, classes, out_path):
    accs   = [class_acc[c] for c in classes]
    colors = ['#ef4444' if a < 0.85 else '#f59e0b' if a < 0.95 else '#22c55e'
              for a in accs]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(classes, accs, color=colors, edgecolor='white', linewidth=0.5)

    ax.axhline(1.00, color='#94a3b8', linewidth=0.8, linestyle=':')
    ax.axhline(0.95, color='#f59e0b', linewidth=1.2, linestyle='--', label='95% line')
    ax.axhline(0.85, color='#ef4444', linewidth=1.2, linestyle='--', label='85% threshold')

    ax.set_ylim(0, 1.09)
    ax.set_xlabel('Letter', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy — ASL Fingerspelling', fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.legend(fontsize=9)

    # Value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Mark similar pairs
    similar_letters = {l for pair in SIMILAR_PAIRS for l in pair}
    for i, cls in enumerate(classes):
        if cls in similar_letters:
            ax.text(i, -0.07, '⚡', ha='center', fontsize=10,
                    transform=ax.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── Plot 3: Presentation dashboard ────────────────────────────────────────

def plot_dashboard(y_true, y_pred, classes, class_acc, mean_acc, std_acc, out_path):
    """Single-slide summary — ideal for a presentation screenshot."""
    fig = plt.figure(figsize=(18, 10), facecolor='#0f172a')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.06, right=0.97,
                            top=0.90, bottom=0.08,
                            wspace=0.35, hspace=0.45)

    DARK  = '#0f172a'
    CARD  = '#1e293b'
    TEXT  = '#f1f5f9'
    GREEN = '#22c55e'
    AMBER = '#f59e0b'
    RED   = '#ef4444'
    INDIGO= '#6366f1'

    # ── Title ──
    fig.text(0.5, 0.95, 'ASL Fingerspelling — Model Evaluation',
             ha='center', va='center', fontsize=18, fontweight='bold',
             color=TEXT, family='monospace')

    # ── KPI cards (top row, span all 3 cols as 3 small axes) ──
    kpi_data = [
        ('CV Accuracy',  f'{mean_acc:.1%}',  f'± {std_acc:.1%}', GREEN),
        ('Classes',      '26',               'A – Z',             INDIGO),
        ('Samples',      str(len(y_true)),   'across 5 folds',    AMBER),
    ]
    for col, (label, big, small, color) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(CARD)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.72, big,   ha='center', fontsize=28, fontweight='bold',
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.40, label, ha='center', fontsize=11, color=TEXT,
                transform=ax.transAxes)
        ax.text(0.5, 0.18, small, ha='center', fontsize=9,  color='#94a3b8',
                transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── Per-class bar (bottom left, span 2 cols) ──
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor(CARD)
    accs   = [class_acc[c] for c in classes]
    colors = [RED if a < 0.85 else AMBER if a < 0.95 else GREEN for a in accs]
    ax2.bar(classes, accs, color=colors, width=0.7)
    ax2.axhline(0.95, color=AMBER, linewidth=1, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 1.12)
    ax2.set_facecolor(CARD)
    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.set_title('Per-Letter Accuracy', color=TEXT, fontsize=11, pad=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax2.spines['bottom'].set_color('#334155')
    ax2.spines['left'].set_color('#334155')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Confused pairs table (bottom right) ──
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(CARD)
    ax3.axis('off')
    ax3.set_title('Similar-Sign Pairs', color=TEXT, fontsize=11, pad=8)

    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    rows = []
    for a, b in SIMILAR_PAIRS:
        if a in classes and b in classes:
            ai, bi = classes.index(a), classes.index(b)
            err = max(cm_norm[ai, bi], cm_norm[bi, ai])
            rows.append((a, b, err))
    rows.sort(key=lambda r: -r[2])

    for i, (a, b, err) in enumerate(rows):
        y_pos = 0.85 - i * 0.14
        color = RED if err > 0.10 else AMBER if err > 0.03 else GREEN
        ax3.text(0.08, y_pos, f'{a}  ↔  {b}',
                 fontsize=13, fontweight='bold', color=TEXT,
                 transform=ax3.transAxes, va='center')
        ax3.text(0.62, y_pos, f'{err:.0%} confused',
                 fontsize=10, color=color,
                 transform=ax3.transAxes, va='center')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model, scaler, classes, X, y = load_artifacts()

    # Cross-validated predictions
    y_true, y_pred, mean_acc, std_acc = cross_val_evaluate(
        model, scaler, classes, X, y, n_splits=5)

    # Per-class accuracy
    class_acc = per_class_accuracy(y_true, y_pred, classes)
    print_per_class_table(class_acc, classes)

    # Worst confusions
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    np.fill_diagonal(cm_norm, 0)
    flat = cm_norm.ravel()
    top5 = np.argsort(flat)[::-1][:5]
    print("\n  Top confusions (off-diagonal):")
    for idx in top5:
        row, col = divmod(idx, len(classes))
        if cm_norm[row, col] > 0:
            print(f"    {classes[row]} misread as {classes[col]}: "
                  f"{cm_norm[row, col]:.0%} of true {classes[row]} samples")

    # Plots
    print("\n[PLOTS] Generating evaluation visuals...")
    plot_confusion_matrix(y_true, y_pred, classes,
                          REPORT_DIR / 'eval_confusion_matrix.png')
    plot_per_class_accuracy(class_acc, classes,
                            REPORT_DIR / 'eval_per_class_accuracy.png')
    plot_dashboard(y_true, y_pred, classes, class_acc, mean_acc, std_acc,
                   REPORT_DIR / 'eval_dashboard.png')

    print(f"\n✓ Done — open training_data/reports/ for all charts.")
    print(f"  👉 Show  eval_dashboard.png  during your presentation.")


if __name__ == '__main__':
    main()
