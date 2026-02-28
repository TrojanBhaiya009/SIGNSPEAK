"""
phrase_recognizer.py
====================
"""

import time
import numpy as np
from collections import deque

# ── Landmark indices ─────────────────────────────────────────────────────────
_LM = {
    'WRIST': 0,
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
    'INDEX_MCP': 5, 'INDEX_PIP': 6, 'INDEX_DIP': 7, 'INDEX_TIP': 8,
    'MIDDLE_MCP': 9, 'MIDDLE_PIP': 10, 'MIDDLE_DIP': 11, 'MIDDLE_TIP': 12,
    'RING_MCP': 13, 'RING_PIP': 14, 'RING_DIP': 15, 'RING_TIP': 16,
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20,
}

DIGIT_CONF_THRESHOLD = 0.75
PHRASE_CONF_THRESHOLD = 0.72
DIGIT_COOLDOWN       = 1.2
PHRASE_COOLDOWN      = 1.5


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _pt(features, landmark_id):
    i = landmark_id * 3
    return np.array(features[i:i + 3], dtype=np.float64)


def _dist(a, b):
    return np.linalg.norm(a - b)


def _tip_above_mcp(features, tip_id, mcp_id, margin=0.0):
    """True if finger tip is above MCP (finger pointing up = extended)."""
    tip_y = _pt(features, tip_id)[1]
    mcp_y = _pt(features, mcp_id)[1]
    return tip_y < mcp_y - margin   # y negative = higher on screen


def _tip_below_mcp(features, tip_id, mcp_id, margin=0.0):
    """True if finger tip is below/level with MCP (finger curled down)."""
    tip_y = _pt(features, tip_id)[1]
    mcp_y = _pt(features, mcp_id)[1]
    return tip_y > mcp_y - margin


def _finger_up(features, tip_id, mcp_id):
    """Extended: tip clearly above MCP."""
    return _tip_above_mcp(features, tip_id, mcp_id, margin=0.05)


def _finger_curled(features, tip_id, mcp_id):
    """Curled: tip at or below MCP level."""
    return _tip_below_mcp(features, tip_id, mcp_id, margin=-0.05)


def _thumb_out(features):
    """Thumb extended sideways: tip far from index MCP."""
    thumb_tip = _pt(features, _LM['THUMB_TIP'])
    index_mcp = _pt(features, _LM['INDEX_MCP'])
    pinky_mcp = _pt(features, _LM['PINKY_MCP'])
    palm_w    = _dist(index_mcp, pinky_mcp)
    if palm_w < 1e-6:
        return False
    d = _dist(thumb_tip, index_mcp)
    return d > palm_w * 0.6


def _thumb_tucked(features):
    """Thumb tucked across palm: tip close to middle/ring MCP."""
    thumb_tip   = _pt(features, _LM['THUMB_TIP'])
    middle_mcp  = _pt(features, _LM['MIDDLE_MCP'])
    ring_mcp    = _pt(features, _LM['RING_MCP'])
    palm_center = (middle_mcp + ring_mcp) / 2
    return _dist(thumb_tip, palm_center) < 0.35


# Shortcuts for finger state checks
def _idx_up(f):  return _finger_up(f,     _LM['INDEX_TIP'],  _LM['INDEX_MCP'])
def _mid_up(f):  return _finger_up(f,     _LM['MIDDLE_TIP'], _LM['MIDDLE_MCP'])
def _rng_up(f):  return _finger_up(f,     _LM['RING_TIP'],   _LM['RING_MCP'])
def _pnk_up(f):  return _finger_up(f,     _LM['PINKY_TIP'],  _LM['PINKY_MCP'])
def _idx_dn(f):  return _finger_curled(f, _LM['INDEX_TIP'],  _LM['INDEX_MCP'])
def _mid_dn(f):  return _finger_curled(f, _LM['MIDDLE_TIP'], _LM['MIDDLE_MCP'])
def _rng_dn(f):  return _finger_curled(f, _LM['RING_TIP'],   _LM['RING_MCP'])
def _pnk_dn(f):  return _finger_curled(f, _LM['PINKY_TIP'],  _LM['PINKY_MCP'])


# ════════════════════════════════════════════════════════════════════════════
# ASL DIGIT detectors  (0-9)
# Uses Y-coordinate comparison — robust for real-world hands.
# ════════════════════════════════════════════════════════════════════════════

def _detect_digit_0(features, fingers):
    """0 — Rounded O: all fingers curled, index tip close to thumb tip."""
    if _idx_dn(features) and _mid_dn(features) and _rng_dn(features) and _pnk_dn(features):
        d = _dist(_pt(features, _LM['THUMB_TIP']), _pt(features, _LM['INDEX_TIP']))
        if d < 0.30:
            return ('0', 0.80)
    return None


def _detect_digit_1(features, fingers):
    """1 — Only index finger pointing up. Middle/ring/pinky all curled down."""
    if (_idx_up(features) and
            _mid_dn(features) and
            _rng_dn(features) and
            _pnk_dn(features)):
        return ('1', 0.82)
    return None


def _detect_digit_3(features, fingers):
    """3 — Thumb OUT + index + middle up, ring + pinky curled.
    Checked before digit 2 — superset of index+middle pattern."""
    if (_idx_up(features) and
            _mid_up(features) and
            _rng_dn(features) and
            _pnk_dn(features) and
            _thumb_out(features)):
        return ('3', 0.80)
    return None


def _detect_digit_2(features, fingers):
    """2 — Index + middle up (V / peace sign), ring + pinky curled.
    Thumb must NOT be out (else it's digit 3)."""
    if (_idx_up(features) and
            _mid_up(features) and
            _rng_dn(features) and
            _pnk_dn(features) and
            not _thumb_out(features)):
        d = _dist(_pt(features, _LM['INDEX_TIP']), _pt(features, _LM['MIDDLE_TIP']))
        if d > 0.05:
            return ('2', 0.80)
    return None


def _detect_digit_4(features, fingers):
    """4 — All four fingers up, thumb tucked."""
    if (_idx_up(features) and
            _mid_up(features) and
            _rng_up(features) and
            _pnk_up(features) and
            _thumb_tucked(features)):
        return ('4', 0.82)
    return None


def _detect_digit_5(features, fingers):
    """5 — All five fingers spread wide open."""
    if (_idx_up(features) and
            _mid_up(features) and
            _rng_up(features) and
            _pnk_up(features) and
            _thumb_out(features)):
        d = _dist(_pt(features, _LM['INDEX_TIP']), _pt(features, _LM['PINKY_TIP']))
        if d > 0.25:
            return ('5', 0.82)
    return None


def _detect_digit_6(features, fingers):
    """6 — Pinky + thumb touching, index/middle/ring extended up."""
    if (_idx_up(features) and _mid_up(features) and _rng_up(features) and _pnk_dn(features)):
        d = _dist(_pt(features, _LM['PINKY_TIP']), _pt(features, _LM['THUMB_TIP']))
        if d < 0.22:
            return ('6', 0.78)
    return None


def _detect_digit_7(features, fingers):
    """7 — Middle + thumb touching, index/ring/pinky extended up."""
    if (_idx_up(features) and _mid_dn(features) and _rng_up(features) and _pnk_up(features)):
        d = _dist(_pt(features, _LM['MIDDLE_TIP']), _pt(features, _LM['THUMB_TIP']))
        if d < 0.22:
            return ('7', 0.78)
    return None


def _detect_digit_8(features, fingers):
    """8 — Index + thumb touching (pinch), middle/ring/pinky extended."""
    if (_idx_dn(features) and _mid_up(features) and _rng_up(features) and _pnk_up(features)):
        d = _dist(_pt(features, _LM['INDEX_TIP']), _pt(features, _LM['THUMB_TIP']))
        if d < 0.22:
            return ('8', 0.80)
    return None


def _detect_digit_9(features, fingers):
    """9 — Index hooked to thumb, middle/ring/pinky extended.
    Lower bound 0.22 separates from digit 8 (d < 0.22)."""
    if (_idx_dn(features) and _mid_up(features) and _rng_up(features) and _pnk_up(features)):
        d = _dist(_pt(features, _LM['INDEX_TIP']), _pt(features, _LM['THUMB_TIP']))
        if 0.22 <= d < 0.28:
            return ('9', 0.78)
    return None


# NOTE: _detect_digit_3 is listed BEFORE _detect_digit_2 intentionally.
DIGIT_CHECKS = [
    _detect_digit_0,
    _detect_digit_1,
    _detect_digit_3,   # ← before 2 (superset check first)
    _detect_digit_2,
    _detect_digit_4,
    _detect_digit_5,
    _detect_digit_6,
    _detect_digit_7,
    _detect_digit_8,
    _detect_digit_9,
]


# ════════════════════════════════════════════════════════════════════════════
# ASL PHRASE / WORD detectors
# Each returns (word_str, confidence) or None.
# These use stricter multi-condition checks to avoid false positives.
# ════════════════════════════════════════════════════════════════════════════

def _palm_facing_camera(features):
    """Rough check: palm faces camera when z of middle_mcp < z of wrist."""
    wrist_z = _pt(features, _LM['WRIST'])[2]
    mid_mcp_z = _pt(features, _LM['MIDDLE_MCP'])[2]
    return mid_mcp_z < wrist_z


def _hand_vertical(features):
    """Hand roughly vertical: middle tip well above wrist."""
    wrist_y = _pt(features, _LM['WRIST'])[1]
    mid_tip_y = _pt(features, _LM['MIDDLE_TIP'])[1]
    return mid_tip_y < wrist_y - 0.15


def _fingers_spread(features):
    """Check that fingers are spread apart (not bunched together)."""
    idx_tip = _pt(features, _LM['INDEX_TIP'])
    mid_tip = _pt(features, _LM['MIDDLE_TIP'])
    rng_tip = _pt(features, _LM['RING_TIP'])
    pnk_tip = _pt(features, _LM['PINKY_TIP'])
    d1 = _dist(idx_tip, mid_tip)
    d2 = _dist(mid_tip, rng_tip)
    d3 = _dist(rng_tip, pnk_tip)
    return d1 > 0.04 and d2 > 0.04 and d3 > 0.04


def _fingers_together(features):
    """Check that four fingers are close together (not spread)."""
    idx_tip = _pt(features, _LM['INDEX_TIP'])
    mid_tip = _pt(features, _LM['MIDDLE_TIP'])
    rng_tip = _pt(features, _LM['RING_TIP'])
    pnk_tip = _pt(features, _LM['PINKY_TIP'])
    d1 = _dist(idx_tip, mid_tip)
    d2 = _dist(mid_tip, rng_tip)
    d3 = _dist(rng_tip, pnk_tip)
    return d1 < 0.12 and d2 < 0.12 and d3 < 0.12


def _all_fingers_curled(features):
    """All four fingers curled into fist."""
    return (_idx_dn(features) and _mid_dn(features) and
            _rng_dn(features) and _pnk_dn(features))


def _detect_phrase_hello(features, fingers):
    """HELLO — Open hand wave: all 5 fingers extended + spread, palm forward,
    hand vertical. Distinguished from digit 5 by requiring fingers together
    (not maximally spread) and palm facing camera."""
    if (_idx_up(features) and _mid_up(features) and
            _rng_up(features) and _pnk_up(features) and
            _thumb_out(features) and
            _palm_facing_camera(features) and
            _hand_vertical(features) and
            _fingers_spread(features)):
        # Extra: finger spread must be moderate (not max like 5)
        idx_tip = _pt(features, _LM['INDEX_TIP'])
        pnk_tip = _pt(features, _LM['PINKY_TIP'])
        spread = _dist(idx_tip, pnk_tip)
        # Hello wave has moderate spread; digit 5 is max spread > 0.40
        if 0.15 < spread < 0.40:
            return ('HELLO', 0.78)
    return None


def _detect_phrase_stop(features, fingers):
    """STOP — Flat hand: all four fingers up + together, thumb out,
    palm facing camera. Key: fingers NOT spread (held together)."""
    if (_idx_up(features) and _mid_up(features) and
            _rng_up(features) and _pnk_up(features) and
            _palm_facing_camera(features) and
            _hand_vertical(features) and
            _fingers_together(features)):
        return ('STOP', 0.80)
    return None


def _detect_phrase_i_love_you(features, fingers):
    """I LOVE YOU — Thumb + index + pinky extended, middle + ring curled.
    The classic ILY handshape. Very distinctive — low false positive risk."""
    if (_idx_up(features) and _pnk_up(features) and
            _mid_dn(features) and _rng_dn(features) and
            _thumb_out(features)):
        # Verify index and pinky are well separated
        idx_tip = _pt(features, _LM['INDEX_TIP'])
        pnk_tip = _pt(features, _LM['PINKY_TIP'])
        separation = _dist(idx_tip, pnk_tip)
        if separation > 0.20:
            return ('I LOVE YOU', 0.85)
    return None


def _detect_phrase_no(features, fingers):
    """NO — Index + middle extended, snap together toward thumb tip.
    Like an alligator mouth closing. Tips of index+middle close to thumb tip."""
    if (_idx_up(features) and _mid_up(features) and
            _rng_dn(features) and _pnk_dn(features)):
        idx_tip = _pt(features, _LM['INDEX_TIP'])
        mid_tip = _pt(features, _LM['MIDDLE_TIP'])
        thm_tip = _pt(features, _LM['THUMB_TIP'])
        # Both fingertips close to thumb = snapping gesture
        d_idx = _dist(idx_tip, thm_tip)
        d_mid = _dist(mid_tip, thm_tip)
        if d_idx < 0.15 and d_mid < 0.15:
            return ('NO', 0.80)
    return None


def _detect_phrase_yes(features, fingers):
    """YES — Closed fist (all fingers curled), thumb may be out or tucked.
    The ASL yes is a fist nodding, detected here as fist shape.
    Distinguished from digit 0 by NOT requiring index-thumb pinch."""
    if _all_fingers_curled(features):
        # Make sure it is NOT digit 0 (index tip touching thumb tip)
        d = _dist(_pt(features, _LM['THUMB_TIP']),
                  _pt(features, _LM['INDEX_TIP']))
        if d > 0.12:
            # Thumb should be tucked or resting alongside fist
            thumb_tip = _pt(features, _LM['THUMB_TIP'])
            index_mcp = _pt(features, _LM['INDEX_MCP'])
            if _dist(thumb_tip, index_mcp) < 0.30:
                return ('YES', 0.76)
    return None


def _detect_phrase_bye(features, fingers):
    """BYE — Open palm facing camera, fingers up + together (like waving bye).
    Similar to STOP but slightly less strict on finger togetherness."""
    if (_idx_up(features) and _mid_up(features) and
            _rng_up(features) and _pnk_up(features) and
            _palm_facing_camera(features) and
            _hand_vertical(features)):
        # Fingers moderately together (not as strict as STOP)
        idx_tip = _pt(features, _LM['INDEX_TIP'])
        pnk_tip = _pt(features, _LM['PINKY_TIP'])
        spread = _dist(idx_tip, pnk_tip)
        # Bye: thumb tucked + fingers medium spread
        if not _thumb_out(features) and 0.10 < spread < 0.30:
            return ('BYE', 0.76)
    return None


# Phrase checks — ordered by specificity (most unique shapes first)
PHRASE_CHECKS = [
    _detect_phrase_i_love_you,   # very distinctive — check first
    _detect_phrase_no,           # index+middle pinch to thumb
    _detect_phrase_yes,          # fist shape
    _detect_phrase_stop,         # flat together hand
    _detect_phrase_hello,        # spread open hand
    _detect_phrase_bye,          # open palm, thumb in
]


# ════════════════════════════════════════════════════════════════════════════
# PhraseRecognizer — digits (0-9) + words (HELLO, STOP, etc.)
# ════════════════════════════════════════════════════════════════════════════

class PhraseRecognizer:
    """
    Detects ASL digits (0-9) and common word signs geometrically from
    hand landmarks (63-dim normalised array).

    Priority order:
      1. Phrase / word signs (HELLO, STOP, I LOVE YOU, NO, YES, BYE)
         — checked first because they are multi-letter tokens and must
           not be split into individual letters.
      2. Digit signs (0-9)
         — geometric detection overrides the MLP for digits.

    Phrases and digits use separate cooldown timers so they never
    interfere with each other.
    """

    DIGIT_SIGNS  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    PHRASE_SIGNS = ['HELLO', 'STOP', 'I LOVE YOU', 'NO', 'YES', 'BYE']
    SIGNS        = PHRASE_SIGNS + DIGIT_SIGNS

    def __init__(self):
        self._last_digit       = None
        self._last_digit_time  = 0
        self._last_phrase      = None
        self._last_phrase_time = 0

    def recognize(self, features):
        """
        Returns (sign_str, confidence) or (None, 0.0).
        Phrase signs are returned as uppercase words (e.g. 'HELLO').
        Digit signs are returned as single-char strings ('0'-'9').
        """
        if features is None or len(features) != 63:
            return None, 0.0

        features = np.array(features, dtype=np.float64)
        fingers  = {}

        # ── Step 1: try phrase / word detectors ──────────────────────
        best_phrase = None
        best_phrase_conf = 0.0
        for check in PHRASE_CHECKS:
            result = check(features, fingers)
            if result and result[1] > best_phrase_conf:
                best_phrase, best_phrase_conf = result

        if best_phrase and best_phrase_conf >= PHRASE_CONF_THRESHOLD:
            return self._emit_phrase(best_phrase, best_phrase_conf)

        # ── Step 2: try digit detectors ──────────────────────────────
        best_digit = None
        best_digit_conf = 0.0
        for check in DIGIT_CHECKS:
            result = check(features, fingers)
            if result and result[1] > best_digit_conf:
                best_digit, best_digit_conf = result

        if best_digit and best_digit_conf >= DIGIT_CONF_THRESHOLD:
            return self._emit_digit(best_digit, best_digit_conf)

        return None, 0.0

    def _emit_digit(self, sign, conf):
        """Apply per-digit cooldown."""
        now = time.time()
        if (sign == self._last_digit and
                now - self._last_digit_time < DIGIT_COOLDOWN):
            return sign, conf

        self._last_digit      = sign
        self._last_digit_time = now
        return sign, conf

    def _emit_phrase(self, sign, conf):
        """Apply per-phrase cooldown (separate from digits)."""
        now = time.time()
        if (sign == self._last_phrase and
                now - self._last_phrase_time < PHRASE_COOLDOWN):
            return sign, conf

        self._last_phrase      = sign
        self._last_phrase_time = now
        return sign, conf

    def reset(self):
        self._last_digit       = None
        self._last_digit_time  = 0
        self._last_phrase      = None
        self._last_phrase_time = 0