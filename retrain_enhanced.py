"""
retrain_enhanced.py
===================
Enhanced ASL alphabet retraining with:
  1. Richer synthetic landmark data (500 samples/letter, multiple pose variants)
  2. Data augmentation (rotation, scaling, mirroring, Gaussian noise)
  3. Feature engineering (inter-finger angles + distances)
  4. Hyperparameter tuning via GridSearchCV
  5. Merges existing real webcam data if available

Usage:
  python retrain_enhanced.py              # retrain with defaults
  python retrain_enhanced.py --quick      # fast mode (skip hyperparameter search)
  python retrain_enhanced.py --evaluate   # also run evaluate_model.py at the end
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_DIR   = Path('training_data')
MODEL_PATH = Path('backend/sign_model.pkl')
REPORT_DIR = Path('training_data/reports')

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# ══════════════════════════════════════════════════════════════════════════
#  1. IMPROVED SYNTHETIC LANDMARK GENERATION
# ══════════════════════════════════════════════════════════════════════════

# Landmark indices for key joints (each landmark has 3 values: x,y,z)
# Index = landmark_number * 3
IDX = {
    'wrist':      slice(0, 3),
    'thumb_cmc':  slice(3, 6),
    'thumb_mcp':  slice(6, 9),
    'thumb_ip':   slice(9, 12),
    'thumb_tip':  slice(12, 15),
    'index_mcp':  slice(15, 18),
    'index_pip':  slice(18, 21),
    'index_dip':  slice(21, 24),
    'index_tip':  slice(24, 27),
    'middle_mcp': slice(27, 30),
    'middle_pip': slice(30, 33),
    'middle_dip': slice(33, 36),
    'middle_tip': slice(36, 39),
    'ring_mcp':   slice(39, 42),
    'ring_pip':   slice(42, 45),
    'ring_dip':   slice(45, 48),
    'ring_tip':   slice(48, 51),
    'pinky_mcp':  slice(51, 54),
    'pinky_pip':  slice(54, 57),
    'pinky_dip':  slice(57, 60),
    'pinky_tip':  slice(60, 63),
}


def _set(lm, name, xyz):
    """Set a named landmark to (x, y, z)."""
    lm[IDX[name]] = xyz


def _base_hand():
    """Return a neutral base hand with all joints in anatomically plausible positions."""
    lm = np.zeros(63, dtype=np.float64)

    # Wrist
    _set(lm, 'wrist',      [0.50, 0.85, 0.00])

    # Thumb chain
    _set(lm, 'thumb_cmc',  [0.42, 0.75, 0.00])
    _set(lm, 'thumb_mcp',  [0.36, 0.68, 0.00])
    _set(lm, 'thumb_ip',   [0.33, 0.60, 0.00])
    _set(lm, 'thumb_tip',  [0.30, 0.55, 0.00])

    # Index chain
    _set(lm, 'index_mcp',  [0.38, 0.55, 0.00])
    _set(lm, 'index_pip',  [0.38, 0.42, 0.00])
    _set(lm, 'index_dip',  [0.38, 0.32, 0.00])
    _set(lm, 'index_tip',  [0.38, 0.22, 0.00])

    # Middle chain
    _set(lm, 'middle_mcp', [0.48, 0.52, 0.00])
    _set(lm, 'middle_pip', [0.48, 0.38, 0.00])
    _set(lm, 'middle_dip', [0.48, 0.28, 0.00])
    _set(lm, 'middle_tip', [0.48, 0.18, 0.00])

    # Ring chain
    _set(lm, 'ring_mcp',   [0.56, 0.54, 0.00])
    _set(lm, 'ring_pip',   [0.56, 0.42, 0.00])
    _set(lm, 'ring_dip',   [0.56, 0.34, 0.00])
    _set(lm, 'ring_tip',   [0.56, 0.26, 0.00])

    # Pinky chain
    _set(lm, 'pinky_mcp',  [0.63, 0.58, 0.00])
    _set(lm, 'pinky_pip',  [0.63, 0.48, 0.00])
    _set(lm, 'pinky_dip',  [0.63, 0.40, 0.00])
    _set(lm, 'pinky_tip',  [0.63, 0.34, 0.00])

    return lm


def _curl_finger(lm, finger, curl_amount=0.8):
    """Curl a finger by moving intermediate joints closer to the base joint.
    Thumb uses cmc/mcp/ip/tip; other fingers use mcp/pip/dip/tip.
    """
    if finger == 'thumb':
        base = lm[IDX['thumb_cmc']].copy()
        for joint in ['mcp', 'ip', 'tip']:
            key = f'thumb_{joint}'
            pos = lm[IDX[key]].copy()
            lm[IDX[key]] = base + (pos - base) * (1 - curl_amount)
    else:
        mcp = lm[IDX[f'{finger}_mcp']].copy()
        for joint in ['pip', 'dip', 'tip']:
            key = f'{finger}_{joint}'
            pos = lm[IDX[key]].copy()
            lm[IDX[key]] = mcp + (pos - mcp) * (1 - curl_amount)


def _extend_finger(lm, finger, direction='up'):
    """Extend a finger fully in a direction.
    Thumb uses cmc/mcp/ip/tip; other fingers use mcp/pip/dip/tip.
    """
    if finger == 'thumb':
        base = lm[IDX['thumb_cmc']].copy()
        if direction == 'up':
            offsets = {'mcp': [0, -0.08, 0], 'ip': [0, -0.18, 0], 'tip': [0, -0.28, 0]}
        elif direction == 'side':
            offsets = {'mcp': [-0.08, -0.04, 0], 'ip': [-0.16, -0.06, 0], 'tip': [-0.24, -0.08, 0]}
        elif direction == 'down':
            offsets = {'mcp': [0, 0.08, 0], 'ip': [0, 0.18, 0], 'tip': [0, 0.28, 0]}
        else:
            offsets = {'mcp': [0, -0.08, 0], 'ip': [0, -0.18, 0], 'tip': [0, -0.28, 0]}
        for joint, off in offsets.items():
            lm[IDX[f'thumb_{joint}']] = base + np.array(off)
    else:
        mcp = lm[IDX[f'{finger}_mcp']].copy()
        if direction == 'up':
            offsets = {'pip': [0, -0.12, 0], 'dip': [0, -0.24, 0], 'tip': [0, -0.36, 0]}
        elif direction == 'side':
            offsets = {'pip': [-0.10, -0.05, 0], 'dip': [-0.20, -0.08, 0], 'tip': [-0.30, -0.10, 0]}
        elif direction == 'down':
            offsets = {'pip': [0, 0.12, 0], 'dip': [0, 0.24, 0], 'tip': [0, 0.36, 0]}
        else:
            offsets = {'pip': [0, -0.12, 0], 'dip': [0, -0.24, 0], 'tip': [0, -0.36, 0]}
        for joint, off in offsets.items():
            lm[IDX[f'{finger}_{joint}']] = mcp + np.array(off)


def _make_fist(lm):
    for f in ['index', 'middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.85)
    _curl_finger(lm, 'thumb', 0.6)
    return lm


# ── Letter-specific templates ─────────────────────────────────────────────

def _template_A(lm):
    """Fist with thumb beside index."""
    _make_fist(lm)
    _set(lm, 'thumb_tip', [0.34, 0.58, 0.02])
    _set(lm, 'thumb_ip',  [0.35, 0.64, 0.01])

def _template_B(lm):
    """Flat hand, fingers up, thumb tucked across palm."""
    for f in ['index', 'middle', 'ring', 'pinky']:
        _extend_finger(lm, f, 'up')
    _curl_finger(lm, 'thumb', 0.7)
    _set(lm, 'thumb_tip', [0.42, 0.62, 0.02])

def _template_C(lm):
    """Curved C-shape — wider opening than O."""
    _set(lm, 'thumb_tip',  [0.30, 0.50, 0.00])
    _set(lm, 'thumb_ip',   [0.33, 0.58, 0.00])
    _set(lm, 'index_tip',  [0.36, 0.32, 0.00])
    _set(lm, 'index_dip',  [0.37, 0.38, 0.00])
    _set(lm, 'middle_tip', [0.44, 0.30, 0.00])
    _set(lm, 'middle_dip', [0.45, 0.36, 0.00])
    _set(lm, 'ring_tip',   [0.52, 0.32, 0.00])
    _set(lm, 'ring_dip',   [0.53, 0.38, 0.00])
    _set(lm, 'pinky_tip',  [0.58, 0.38, 0.00])
    _set(lm, 'pinky_dip',  [0.59, 0.42, 0.00])

def _template_D(lm):
    """Index pointing up, others form circle with thumb."""
    _extend_finger(lm, 'index', 'up')
    for f in ['middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.8)
    _set(lm, 'thumb_tip', [0.44, 0.48, 0.02])
    _set(lm, 'middle_tip', [0.46, 0.50, 0.01])

def _template_E(lm):
    """All fingers curled, thumb across fingertips."""
    for f in ['index', 'middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.75)
    _set(lm, 'thumb_tip', [0.40, 0.48, 0.03])
    _set(lm, 'thumb_ip',  [0.38, 0.55, 0.02])

def _template_F(lm):
    """Index+thumb circle, middle/ring/pinky up."""
    _set(lm, 'thumb_tip', [0.37, 0.50, 0.02])
    _set(lm, 'index_tip', [0.39, 0.50, 0.01])
    _curl_finger(lm, 'index', 0.6)
    for f in ['middle', 'ring', 'pinky']:
        _extend_finger(lm, f, 'up')

def _template_G(lm):
    """Index + thumb pointing sideways."""
    _extend_finger(lm, 'index', 'side')
    _set(lm, 'thumb_tip', [0.25, 0.58, 0.00])
    _set(lm, 'thumb_ip',  [0.30, 0.62, 0.00])
    for f in ['middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.85)

def _template_H(lm):
    """Index + middle pointing sideways."""
    _extend_finger(lm, 'index', 'side')
    _extend_finger(lm, 'middle', 'side')
    for f in ['ring', 'pinky']:
        _curl_finger(lm, f, 0.85)
    _curl_finger(lm, 'thumb', 0.6)

def _template_I(lm):
    """Pinky up, rest fist."""
    _make_fist(lm)
    _extend_finger(lm, 'pinky', 'up')

def _template_J(lm):
    """Like I but pinky tilted/rotated (J-draw)."""
    _make_fist(lm)
    _extend_finger(lm, 'pinky', 'up')
    _set(lm, 'pinky_tip', [0.68, 0.28, 0.02])
    _set(lm, 'wrist', [0.52, 0.83, 0.01])

def _template_K(lm):
    """Index + middle up in V, thumb between them."""
    _extend_finger(lm, 'index', 'up')
    _extend_finger(lm, 'middle', 'up')
    # Spread them apart
    lm[IDX['index_tip']] += np.array([-0.04, 0, 0])
    lm[IDX['middle_tip']] += np.array([0.04, 0, 0])
    _set(lm, 'thumb_tip', [0.42, 0.42, 0.03])
    for f in ['ring', 'pinky']:
        _curl_finger(lm, f, 0.85)

def _template_L(lm):
    """L-shape: index up, thumb out sideways."""
    _extend_finger(lm, 'index', 'up')
    _set(lm, 'thumb_tip', [0.22, 0.60, 0.00])
    _set(lm, 'thumb_ip',  [0.28, 0.64, 0.00])
    for f in ['middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.85)

def _template_M(lm):
    """Thumb under 3 fingers."""
    for f in ['index', 'middle', 'ring']:
        _curl_finger(lm, f, 0.65)
    _curl_finger(lm, 'pinky', 0.85)
    _set(lm, 'thumb_tip', [0.52, 0.66, 0.03])

def _template_N(lm):
    """Thumb under 2 fingers."""
    for f in ['index', 'middle']:
        _curl_finger(lm, f, 0.65)
    for f in ['ring', 'pinky']:
        _curl_finger(lm, f, 0.85)
    _set(lm, 'thumb_tip', [0.50, 0.64, 0.03])

def _template_O(lm):
    """All fingertips touching thumb — tight circle (narrower than C)."""
    _set(lm, 'thumb_tip',  [0.40, 0.45, 0.01])
    _set(lm, 'index_tip',  [0.42, 0.43, 0.00])
    _set(lm, 'middle_tip', [0.44, 0.42, 0.00])
    _set(lm, 'ring_tip',   [0.46, 0.43, 0.00])
    _set(lm, 'pinky_tip',  [0.48, 0.45, 0.00])
    for f in ['index', 'middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.55)

def _template_P(lm):
    """Like K but pointing down."""
    _extend_finger(lm, 'index', 'down')
    _extend_finger(lm, 'middle', 'down')
    _set(lm, 'thumb_tip', [0.42, 0.62, 0.02])
    for f in ['ring', 'pinky']:
        _curl_finger(lm, f, 0.85)

def _template_Q(lm):
    """Like G but pointing down."""
    _extend_finger(lm, 'index', 'down')
    _set(lm, 'thumb_tip', [0.38, 0.78, 0.00])
    for f in ['middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.85)

def _template_R(lm):
    """Crossed index and middle up."""
    _extend_finger(lm, 'index', 'up')
    _extend_finger(lm, 'middle', 'up')
    # Cross: swap x of tips
    lm[IDX['index_tip']][0] = 0.50
    lm[IDX['middle_tip']][0] = 0.40
    lm[IDX['index_dip']][0] = 0.46
    lm[IDX['middle_dip']][0] = 0.43
    for f in ['ring', 'pinky']:
        _curl_finger(lm, f, 0.85)
    _curl_finger(lm, 'thumb', 0.6)

def _template_S(lm):
    """Fist with thumb over curled fingers."""
    _make_fist(lm)
    _set(lm, 'thumb_tip', [0.42, 0.52, 0.04])
    _set(lm, 'thumb_ip',  [0.38, 0.56, 0.03])

def _template_T(lm):
    """Thumb between index and middle."""
    for f in ['index', 'middle', 'ring', 'pinky']:
        _curl_finger(lm, f, 0.75)
    _set(lm, 'thumb_tip', [0.43, 0.50, 0.04])

def _template_U(lm):
    """Index + middle up together, close."""
    _extend_finger(lm, 'index', 'up')
    _extend_finger(lm, 'middle', 'up')
    # Keep them close together
    lm[IDX['index_tip']][0] = 0.42
    lm[IDX['middle_tip']][0] = 0.46
    for f in ['ring', 'pinky']:
        _curl_finger(lm, f, 0.85)
    _curl_finger(lm, 'thumb', 0.6)

def _template_V(lm):
    """Peace/V sign — index + middle spread apart."""
    _extend_finger(lm, 'index', 'up')
    _extend_finger(lm, 'middle', 'up')
    # Spread them wide
    lm[IDX['index_tip']][0] = 0.32
    lm[IDX['middle_tip']][0] = 0.56
    lm[IDX['index_dip']][0] = 0.34
    lm[IDX['middle_dip']][0] = 0.54
    for f in ['ring', 'pinky']:
        _curl_finger(lm, f, 0.85)
    _curl_finger(lm, 'thumb', 0.6)

def _template_W(lm):
    """3 fingers (index, middle, ring) up and spread."""
    for f in ['index', 'middle', 'ring']:
        _extend_finger(lm, f, 'up')
    lm[IDX['index_tip']][0] = 0.32
    lm[IDX['middle_tip']][0] = 0.48
    lm[IDX['ring_tip']][0] = 0.62
    _curl_finger(lm, 'pinky', 0.85)
    _curl_finger(lm, 'thumb', 0.6)

def _template_X(lm):
    """Index bent like a hook, rest fist."""
    _make_fist(lm)
    _set(lm, 'index_pip', [0.38, 0.38, 0.00])
    _set(lm, 'index_dip', [0.42, 0.42, 0.02])
    _set(lm, 'index_tip', [0.44, 0.48, 0.01])

def _template_Y(lm):
    """Thumb + pinky out (hang-loose)."""
    _make_fist(lm)
    _extend_finger(lm, 'pinky', 'up')
    _set(lm, 'thumb_tip', [0.22, 0.58, 0.00])
    _set(lm, 'thumb_ip',  [0.28, 0.62, 0.00])

def _template_Z(lm):
    """Index pointing (static pose for Z)."""
    _make_fist(lm)
    _extend_finger(lm, 'index', 'up')
    lm[IDX['index_tip']][0] += 0.04


TEMPLATES = {
    'A': _template_A, 'B': _template_B, 'C': _template_C, 'D': _template_D,
    'E': _template_E, 'F': _template_F, 'G': _template_G, 'H': _template_H,
    'I': _template_I, 'J': _template_J, 'K': _template_K, 'L': _template_L,
    'M': _template_M, 'N': _template_N, 'O': _template_O, 'P': _template_P,
    'Q': _template_Q, 'R': _template_R, 'S': _template_S, 'T': _template_T,
    'U': _template_U, 'V': _template_V, 'W': _template_W, 'X': _template_X,
    'Y': _template_Y, 'Z': _template_Z,
}


# ══════════════════════════════════════════════════════════════════════════
#  2. DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════

def _rotate_2d(pts_63, angle_deg):
    """Rotate all landmarks around the wrist in the xy-plane."""
    pts = pts_63.copy().reshape(21, 3)
    wrist = pts[0].copy()
    pts -= wrist
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts[:, :2] = pts[:, :2] @ rot.T
    pts += wrist
    return pts.flatten()


def _scale_jitter(pts_63, scale_factor):
    """Scale all landmarks relative to the wrist."""
    pts = pts_63.copy().reshape(21, 3)
    wrist = pts[0].copy()
    pts -= wrist
    pts *= scale_factor
    pts += wrist
    return pts.flatten()


def _mirror_x(pts_63):
    """Mirror landmarks horizontally (simulate left/right hand)."""
    pts = pts_63.copy().reshape(21, 3)
    pts[:, 0] = 1.0 - pts[:, 0]
    return pts.flatten()


def _translate_jitter(pts_63, dx, dy):
    """Shift all landmarks by (dx, dy)."""
    pts = pts_63.copy().reshape(21, 3)
    pts[:, 0] += dx
    pts[:, 1] += dy
    return pts.flatten()


def augment_sample(lm, rng):
    """Apply random augmentations to a single 63-dim landmark vector."""
    result = lm.copy()

    # Random rotation ±15°
    angle = rng.uniform(-15, 15)
    result = _rotate_2d(result, angle)

    # Random scale 0.85–1.15
    scale = rng.uniform(0.85, 1.15)
    result = _scale_jitter(result, scale)

    # Random translation ±0.05
    dx, dy = rng.uniform(-0.05, 0.05, size=2)
    result = _translate_jitter(result, dx, dy)

    # Random mirror (20% chance)
    if rng.random() < 0.20:
        result = _mirror_x(result)

    # Gaussian noise
    result += rng.normal(0, 0.015, size=63)

    return result


# ══════════════════════════════════════════════════════════════════════════
#  3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

def normalize_landmarks(flat_63):
    """Normalize landmarks relative to wrist (position+scale invariant)."""
    pts = np.array(flat_63, dtype=np.float64).reshape(21, 3)
    wrist = pts[0].copy()
    pts -= wrist
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist
    return pts.flatten()


def compute_extra_features(normed_63):
    """
    Compute additional features from normalised landmarks:
      - 10 pairwise fingertip distances (5 tips choose 2)
      - 5 fingertip-to-wrist distances
      - 5 finger curl ratios (tip_y vs mcp_y)
      - 4 inter-finger spread angles
    Total: 24 extra features
    """
    pts = normed_63.reshape(21, 3)
    tips = [4, 8, 12, 16, 20]      # thumb, index, middle, ring, pinky tips
    mcps = [2, 5, 9, 13, 17]       # corresponding MCP-like joints

    features = []

    # Pairwise tip distances (10)
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            d = np.linalg.norm(pts[tips[i]] - pts[tips[j]])
            features.append(d)

    # Tip-to-wrist distances (5)
    for t in tips:
        features.append(np.linalg.norm(pts[t]))

    # Curl ratio: how far is tip from mcp? (5)
    for t, m in zip(tips, mcps):
        mcp_y = pts[m, 1]
        tip_y = pts[t, 1]
        features.append(tip_y - mcp_y)  # negative = extended up

    # Inter-finger spread angles (4, between adjacent tips)
    for i in range(len(tips) - 1):
        v1 = pts[tips[i]] - pts[0]      # vector from wrist to tip1
        v2 = pts[tips[i+1]] - pts[0]    # vector from wrist to tip2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features.append(cos_angle)

    return np.array(features, dtype=np.float64)


def build_features(raw_63):
    """Normalise landmarks and append engineered features → 87-dim vector."""
    normed = normalize_landmarks(raw_63)
    extra  = compute_extra_features(normed)
    return np.concatenate([normed, extra])


# ══════════════════════════════════════════════════════════════════════════
#  4. DATA GENERATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def generate_dataset(samples_per_letter=500, seed=42):
    """Generate synthetic dataset with augmentation."""
    rng = np.random.default_rng(seed)
    X, y = [], []

    print(f"\n[GEN] Generating {samples_per_letter} samples × 26 letters = "
          f"{samples_per_letter * 26} total")

    for letter in ALPHABET:
        template_fn = TEMPLATES[letter]

        # Generate base + variants with different noise levels
        for i in range(samples_per_letter):
            lm = _base_hand()
            template_fn(lm)

            # Add per-joint anatomical jitter (more realistic than global noise)
            joint_jitter = rng.normal(0, 0.012, size=63)
            lm += joint_jitter

            # Apply augmentations
            lm = augment_sample(lm, rng)

            # Build feature vector
            feat = build_features(lm)
            X.append(feat)
            y.append(letter)

        if (ALPHABET.index(letter) + 1) % 5 == 0:
            print(f"  Generated A-{letter}...")

    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    print(f"  ✓ Synthetic data: {X.shape[0]} samples, {X.shape[1]} features each")
    return X, y


def load_existing_data():
    """Load existing real webcam training data if available."""
    x_path = DATA_DIR / 'X_train.npy'
    y_path = DATA_DIR / 'y_train.npy'

    if not x_path.exists() or not y_path.exists():
        return None, None

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"\n[LOAD] Found existing real data: {len(X)} samples, {X.shape[1]} features")

    # If existing data is 63-dim, add engineered features
    if X.shape[1] == 63:
        print("  → Converting 63→87 features (adding engineered features)...")
        X_new = np.array([build_features(row) for row in X])
        return X_new, y
    elif X.shape[1] == 87:
        return X, y
    else:
        print(f"  [WARN] Unexpected feature dim {X.shape[1]}, skipping merge.")
        return None, None


# ══════════════════════════════════════════════════════════════════════════
#  5. TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_with_tuning(X_train, y_train, quick=False):
    """Train MLP with optional hyperparameter tuning via GridSearchCV."""

    if quick:
        print("\n[TRAIN] Quick mode — using default architecture (256,128,64)")
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=42,
            verbose=False,
            learning_rate='adaptive',
            alpha=0.0005,
        )
        model.fit(X_train, y_train)
        return model

    print("\n[TRAIN] Running hyperparameter search (this may take a few minutes)...")
    param_grid = {
        'hidden_layer_sizes': [
            (256, 128, 64),
            (512, 256, 128),
            (256, 128, 64, 32),
            (384, 192, 96),
        ],
        'alpha': [0.0001, 0.0005, 0.001],
        'learning_rate': ['adaptive'],
    }

    base_model = MLPClassifier(
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        base_model, param_grid,
        cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best params: {search.best_params_}")
    print(f"  Best CV accuracy: {search.best_score_:.1%}")

    return search.best_estimator_


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Enhanced ASL Alphabet Retraining')
    parser.add_argument('--quick', action='store_true',
                        help='Skip hyperparameter search, use defaults')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluate_model.py after training')
    parser.add_argument('--samples', type=int, default=500,
                        help='Samples per letter (default: 500)')
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  🤟 Enhanced ASL Alphabet Retraining")
    print("=" * 60)

    t0 = time.time()

    # ── 1. Generate synthetic data ──
    X_syn, y_syn = generate_dataset(samples_per_letter=args.samples)

    # ── 2. Merge with existing real data ──
    X_real, y_real = load_existing_data()
    if X_real is not None:
        print(f"\n[MERGE] Combining {len(X_syn)} synthetic + {len(X_real)} real samples")
        X_all = np.vstack([X_syn, X_real])
        y_all = np.concatenate([y_syn, y_real])
    else:
        X_all, y_all = X_syn, y_syn
        print("\n[MERGE] Using synthetic data only (no existing data found)")

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    print(f"\n[DATA] Final dataset: {len(X_all)} samples, {X_all.shape[1]} features, "
          f"{len(np.unique(y_all))} classes")

    # ── 3. Encode + split ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 4. Train ──
    model = train_with_tuning(X_train_sc, y_train, quick=args.quick)

    train_acc = model.score(X_train_sc, y_train)
    test_acc  = model.score(X_test_sc, y_test)
    gap       = train_acc - test_acc

    print(f"\n{'=' * 50}")
    print(f"  RESULTS")
    print(f"{'=' * 50}")
    print(f"  Train accuracy : {train_acc:.1%}")
    print(f"  Test  accuracy : {test_acc:.1%}")
    print(f"  Overfit gap    : {gap:.1%}", end='')
    if gap > 0.15:
        print("  ⚠  Overfitting")
    elif gap > 0.07:
        print("  ⚠  Mild overfitting")
    else:
        print("  ✓  Healthy")

    y_pred = model.predict(X_test_sc)
    print(f"\n[REPORT] Per-class metrics:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── 5. Save model ──
    model_data = {
        'model':          model,
        'scaler':         scaler,
        'classes':        list(le.classes_),
        'label_encoder':  le,
        'test_accuracy':  test_acc,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[SAVE] Model saved → {MODEL_PATH}")

    # ── 6. Save training data for evaluate_model.py ──
    # IMPORTANT: Save RAW (unscaled) X_train so that train_combined.py /
    # evaluate_model.py can apply their own scaler fresh.
    # Saving X_train_sc (already scaled) caused double-scaling bug — FIXED.
    np.save(DATA_DIR / 'X_train.npy', X_train)        # ← raw, unscaled ✓
    np.save(DATA_DIR / 'y_train.npy', le.inverse_transform(y_train))
    print(f"[SAVE] Training data saved → {DATA_DIR}/")
    print(f"       X_train.npy shape: {X_train.shape}  (raw, unscaled — scaler is inside pkl)")

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Data:  {DATA_DIR}/X_train.npy, y_train.npy")

    # ── 7. Optionally run evaluation ──
    if args.evaluate:
        print(f"\n{'=' * 60}")
        print("  Running evaluation...")
        print(f"{'=' * 60}")
        os.system(f'{sys.executable} evaluate_model.py')


if __name__ == '__main__':
    main()