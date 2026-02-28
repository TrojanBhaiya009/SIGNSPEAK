"""
SignSpeak Sign Language Recognition Model
==========================================
Single-frame ASL fingerspelling classifier — A-Z + 0-9.

"""

import os
import pickle
import time
import numpy as np
from typing import Any

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from phrase_recognizer import PhraseRecognizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.70
LETTER_HOLD_FRAMES   = 8
SAME_LETTER_COOLDOWN = 1.2
DIGIT_COOLDOWN       = 1.2

DIGIT_SIGNS          = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
PHRASE_SIGNS         = {'HELLO', 'STOP', 'I LOVE YOU', 'NO', 'YES', 'BYE'}
DIGIT_CONF_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# MediaPipe setup (lazy-loaded)
# ---------------------------------------------------------------------------

_detector = None

def _get_detector():
    global _detector
    if _detector is not None:
        return _detector
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hand_landmarker.task')
        if not os.path.exists(model_path):
            print(f"[MEDIAPIPE] hand_landmarker.task not found at {model_path}")
            return None

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        _detector = vision.HandLandmarker.create_from_options(options)
        print("[MEDIAPIPE] HandLandmarker initialised.")
        return _detector
    except Exception as e:
        print(f"[MEDIAPIPE] Could not initialise detector: {e}")
        return None


def extract_landmarks(frame):
    if frame is None: return None, None
    if not isinstance(frame, np.ndarray): return None, None
    if frame.ndim != 3 or frame.shape[2] != 3: return None, None
    if frame.size == 0: return None, None

    detector = _get_detector()
    if detector is None: return None, None

    try:
        import mediapipe as mp
        frame_rgb = frame[:, :, ::-1].copy()
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame_rgb))
        results = detector.detect(mp_image)
        if not results.hand_landmarks or len(results.hand_landmarks) == 0:
            return None, None
        hand = results.hand_landmarks[0]
        if len(hand) != 21: return None, None
        raw = []
        for lm in hand:
            raw.extend([lm.x, lm.y, lm.z])
        raw_arr = np.array(raw, dtype=np.float64)
        if not np.all(np.isfinite(raw_arr)): return None, None
        features = normalize_landmarks(raw_arr)
        return features, hand
    except Exception as e:
        print(f"[LANDMARKS] Extraction error: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Landmark normalisation
# ---------------------------------------------------------------------------

def normalize_landmarks(flat_landmarks):
    pts = np.array(flat_landmarks, dtype=np.float64).reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts.flatten()


def compute_extra_features(normed_63):
    pts = normed_63.reshape(21, 3)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    features = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            features.append(np.linalg.norm(pts[tips[i]] - pts[tips[j]]))
    for t in tips:
        features.append(np.linalg.norm(pts[t]))
    for t, m in zip(tips, mcps):
        features.append(pts[t, 1] - pts[m, 1])
    for i in range(len(tips) - 1):
        v1 = pts[tips[i]] - pts[0]
        v2 = pts[tips[i+1]] - pts[0]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features.append(cos_a)
    return np.array(features, dtype=np.float64)


def build_features_87(raw_63):
    normed = normalize_landmarks(raw_63)
    extra  = compute_extra_features(normed)
    return np.concatenate([normed, extra])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SignLanguageModel:

    def __init__(self, model_path='sign_model.pkl', classes_path=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, model_path)

        self.model: Any        = None
        self.scaler: Any       = None
        self.classes      = []
        self.is_trained   = False
        self._expects_87  = False

        self._candidate_letter = None
        self._candidate_count  = 0

        self._caption          = ''
        self._last_letter      = None
        self._last_letter_time = 0

        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self._digit_recognizer    = PhraseRecognizer()
        self._load()

    def _load(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"[MODEL] No trained model found at {self.model_path}")
                return
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.model      = data['model']
            self.scaler     = data['scaler']
            self.classes    = data['classes']
            self.is_trained = True
            n_features = self.scaler.n_features_in_
            self._expects_87 = (n_features == 87)
            acc = data.get('test_accuracy', None)
            acc_str  = f" | test acc: {acc:.1%}" if acc else ""
            feat_str = f" | {n_features}-dim features"
            print(f"[MODEL] Loaded sklearn MLP from {self.model_path}"
                  f" | {len(self.classes)} classes{acc_str}{feat_str}")
            print(f"[MODEL] Classes: {self.classes}")
        except Exception as e:
            print(f"[MODEL] Load error: {e}")
            self.model      = None
            self.is_trained = False

    # ------------------------------------------------------------------
    # Prediction — priority order:
    #   1. Geometric phrase detector  (HELLO, STOP, I LOVE YOU, NO, YES, BYE)
    #   2. Geometric digit detector   (0-9)
    #   3. MLP for letters A-Z (and digits if MLP is very confident)
    # ------------------------------------------------------------------

    def predict_from_landmarks(self, features):
        if not self.is_trained or self.model is None:
            # Even without trained MLP, geometric detection still works
            return self._predict_geometric_only(features)

        try:
            arr = np.array(features, dtype=np.float64)
            if arr.shape[0] != 63:
                return None, 0.0
            if not np.all(np.isfinite(arr)):
                return None, 0.0

            # ── Step 1: geometric phrase + digit recognizer ──────────────
            geo_sign, geo_conf = self._digit_recognizer.recognize(arr)

            if geo_sign and geo_conf >= DIGIT_CONF_THRESHOLD:
                kind = 'PHRASE' if geo_sign in PHRASE_SIGNS else 'DIGIT'
                print(f"[GEO] {kind} {geo_sign} ({geo_conf:.2f})")
                return geo_sign, geo_conf

            # ── Step 2: MLP for letters A-Z (and high-confidence digits) ─
            if self._expects_87:
                feat = build_features_87(arr).reshape(1, -1)
            else:
                feat = normalize_landmarks(arr).reshape(1, -1)

            if feat.shape[1] != self.scaler.n_features_in_:
                return None, 0.0

            scaled     = self.scaler.transform(feat)
            proba      = self.model.predict_proba(scaled)[0]
            confidence = float(np.max(proba))
            idx        = int(np.argmax(proba))
            mlp_sign   = self.classes[idx]

            # If MLP predicts a digit but geometry didn't fire, only accept
            # if MLP is very confident (>= 0.90). Prevents D↔1, V↔2 etc.
            if mlp_sign in DIGIT_SIGNS:
                if confidence >= 0.90:
                    return mlp_sign, confidence
                return None, 0.0

            # Accept MLP letter result at normal confidence threshold
            if confidence >= self.confidence_threshold:
                return mlp_sign, confidence

        except Exception as e:
            print(f"[PREDICT] Error: {e}")

        return None, 0.0

    def _predict_geometric_only(self, features):
        """Fallback when MLP is not trained — still detect phrases + digits."""
        try:
            arr = np.array(features, dtype=np.float64)
            if arr.shape[0] != 63 or not np.all(np.isfinite(arr)):
                return None, 0.0
            geo_sign, geo_conf = self._digit_recognizer.recognize(arr)
            if geo_sign and geo_conf >= DIGIT_CONF_THRESHOLD:
                return geo_sign, geo_conf
        except Exception:
            pass
        return None, 0.0

    # ------------------------------------------------------------------
    # Caption building
    # ------------------------------------------------------------------

    def update_spelling(self, letter, confidence):
        if letter is None:
            self._candidate_count = 0
            return

        is_digit  = letter in DIGIT_SIGNS
        is_phrase = letter in PHRASE_SIGNS

        if letter == self._candidate_letter:
            self._candidate_count += 1
        else:
            self._candidate_letter = letter
            self._candidate_count  = 1

        # Phrases commit at 5 frames, digits at 4, letters at default
        if is_phrase:
            hold_needed = 5
        elif is_digit:
            hold_needed = 4
        else:
            hold_needed = LETTER_HOLD_FRAMES

        if self._candidate_count < hold_needed:
            return

        now      = time.time()
        if is_phrase:
            cooldown = 2.0   # longer cooldown for phrases
        elif is_digit:
            cooldown = DIGIT_COOLDOWN
        else:
            cooldown = SAME_LETTER_COOLDOWN

        if (letter == self._last_letter and
                now - self._last_letter_time < cooldown):
            return

        # Phrases get appended as whole words with a leading space
        if is_phrase:
            if self._caption and not self._caption.endswith(' '):
                self._caption += ' '
            self._caption += letter + ' '
        else:
            self._caption += letter

        self._last_letter       = letter
        self._last_letter_time  = now
        self._candidate_count   = 0

    def get_caption(self):      return self._caption
    def add_space(self):        self._caption += ' '
    def backspace(self):        self._caption = self._caption[:-1]

    def clear_caption(self):
        self._caption          = ''
        self._last_letter      = None
        self._last_letter_time = 0
        self._candidate_letter = None
        self._candidate_count  = 0
        self._digit_recognizer.reset()

    def get_status(self):
        return {
            'is_trained'          : self.is_trained,
            'num_classes'         : len(self.classes),
            'classes'             : self.classes,
            'digit_signs'         : list(DIGIT_SIGNS),
            'phrase_signs'        : list(PHRASE_SIGNS),
            'model_type'          : 'sklearn MLP (A-Z) + geometric (0-9, phrases)'
                                    if self.is_trained else 'geometric only (phrases + digits)',
            'confidence_threshold': self.confidence_threshold,
            'caption'             : self._caption,
        }

    def train(self, X_train, y_train, **kwargs):
        return {
            'success': False,
            'error': (
                'Online training is not supported. '
                'Run convert_json_to_npy.py then train_combined.py offline, '
                'then restart the server.'
            ),
        }