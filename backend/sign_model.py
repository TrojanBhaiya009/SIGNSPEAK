"""
SignSpeak ASL Recognition Model
================================
Fresh untrained model for ASL alphabet recognition.
Uses hand landmarks (21 points x 3 coords = 63 features) from MediaPipe.

Supports:
  - Rule-based detection for clear gesture patterns (fallback)
  - ML-based detection after training (MLPClassifier)
  - Real-time spelling and caption building
  - Data collection and training pipeline
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import pickle
import time
from collections import deque


def normalize_landmarks(flat_landmarks):
    """
    Normalize 63-dim landmarks relative to wrist (landmark 0).
    Makes detection position-invariant and scale-invariant.
    Must match the normalization in train_alphabet.py.
    """
    pts = np.array(flat_landmarks, dtype=np.float64).reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts.flatten()


class SignLanguageModel:
    def __init__(self, model_path='sign_model.pkl'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, model_path)
        self.model = None
        self.scaler = None
        self.classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.is_trained = False

        # Spelling buffer
        self.letter_buffer = []
        self.last_letter = None
        self.last_letter_time = 0
        self.letter_hold_time = 0.5   # seconds to hold a sign before it registers
        self.word_timeout = 2.0       # seconds of no input to finalize a word
        self.current_word = ""
        self.completed_words = []

        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.30

        # Try to load existing model
        self.load_model()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_model(self):
        """Load a previously trained model from disk (if any)."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data.get('model')
                self.scaler = data.get('scaler')
                self.label_encoder = data.get('label_encoder')
                saved_classes = data.get('classes')
                if saved_classes:
                    self.classes = saved_classes
                self.is_trained = self.model is not None and self.scaler is not None
                if self.is_trained:
                    print(f"[MODEL] Loaded trained model -- {len(self.classes)} classes")
            else:
                print("[MODEL] No trained model found -- starting untrained (rule-based only)")
        except Exception as e:
            print(f"[MODEL] Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.is_trained = False

    def save_model(self):
        """Persist model, scaler, and class list to disk."""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes,
            'label_encoder': getattr(self, 'label_encoder', None),
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[MODEL] Saved -- {len(self.classes)} classes")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train, y_train):
        """
        Train from scratch on landmark data.

        Args
        ----
        X_train : np.ndarray, shape (n_samples, 63)
        y_train : np.ndarray, shape (n_samples,)  -- letter labels

        Returns
        -------
        dict with success flag, accuracy, class info
        """
        if len(X_train) == 0:
            return {'error': 'No training data provided', 'success': False}
        if len(X_train) < 5:
            return {'error': 'Need at least 5 samples to train', 'success': False}

        unique_classes = sorted(list(set(y_train)))
        if len(unique_classes) < 2:
            return {'error': 'Need at least 2 different classes to train', 'success': False}

        self.classes = unique_classes

        # Encode string labels to integers (avoids scikit-learn early_stopping bug)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)

        # Fresh model
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )

        X = X_train.astype(np.float64)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_encoded)

        accuracy = self.model.score(X_scaled, y_train)
        self.is_trained = True
        self.save_model()

        print(f"[MODEL] Trained -- {len(X_train)} samples, "
              f"{len(unique_classes)} classes, {accuracy:.1%} acc")

        return {
            'success': True,
            'accuracy': float(accuracy),
            'num_samples': int(len(X_train)),
            'num_classes': len(unique_classes),
            'classes': unique_classes,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_from_landmarks(self, landmarks):
        """
        Predict a letter from 63-dim landmark features.

        Returns (letter, confidence) or (None, 0.0).
        """
        # ML model first (if trained)
        if self.is_trained and self.model is not None and self.scaler is not None:
            try:
                normed = normalize_landmarks(landmarks)
                X = normed.reshape(1, -1).astype(np.float64)
                X_scaled = self.scaler.transform(X)
                proba = self.model.predict_proba(X_scaled)[0]
                confidence = float(np.max(proba))
                letter = self.classes[int(np.argmax(proba))]
                if confidence > self.confidence_threshold:
                    return letter, confidence
            except Exception:
                pass

        # Fallback -- rule-based
        return self._rule_based_predict(landmarks)

    def _rule_based_predict(self, landmarks):
        """Heuristic ASL recognition from finger extension states."""
        if len(landmarks) < 63:
            return None, 0.0

        pts = np.array(landmarks).reshape(21, 3)

        # --- finger extension ---
        thumb = abs(pts[4][0] - pts[0][0]) > 0.12
        index = pts[8][1] < pts[5][1] - 0.05
        middle = pts[12][1] < pts[9][1] - 0.05
        ring = pts[16][1] < pts[13][1] - 0.05
        pinky = pts[20][1] < pts[17][1] - 0.05

        # Clustered fingertips (O, C, E) -- ambiguous, skip
        tips = [pts[4], pts[8], pts[12], pts[16], pts[20]]
        if np.std([p[0] for p in tips]) < 0.08:
            return None, 0.0
        # Thumb-index touch (F/O) -- ambiguous
        if np.linalg.norm(pts[4] - pts[8]) < 0.08:
            return None, 0.0

        i, m, r, p, t = index, middle, ring, pinky, thumb

        if not i and not m and not r and not p:
            return ('A', 0.75) if t else ('S', 0.70)
        if i and not m and not r and not p:
            return ('L', 0.80) if t else ('D', 0.60)
        if i and m and not r and not p:
            btwn = (min(pts[8][0], pts[12][0]) < pts[4][0] < max(pts[8][0], pts[12][0]))
            if btwn and abs(pts[4][1] - (pts[8][1] + pts[12][1]) / 2) < 0.15:
                return 'K', 0.85
            return 'V', 0.70
        if i and m and r and not p:
            return 'W', 0.80
        if not i and not m and not r and p:
            return 'I', 0.80
        if t and not i and not m and not r and p:
            return 'Y', 0.85
        if i and m and r and p and t:
            return 'B', 0.75

        return None, 0.0

    # ------------------------------------------------------------------
    # Spelling / Caption
    # ------------------------------------------------------------------

    def update_spelling(self, letter, confidence):
        """Add letter to spelling buffer when sign is held steady."""
        now = time.time()

        # Finalize word on timeout
        if self.letter_buffer and (now - self.last_letter_time) > self.word_timeout:
            self._finalize_word()

        if letter is None or confidence < self.confidence_threshold:
            return

        if letter == self.last_letter:
            if (now - self.last_letter_time) >= self.letter_hold_time:
                if not self.letter_buffer or self.letter_buffer[-1] != letter:
                    self.letter_buffer.append(letter)
                    self.current_word = ''.join(self.letter_buffer)
                    self.last_letter_time = now
        else:
            self.last_letter = letter
            self.last_letter_time = now

    def _finalize_word(self):
        if self.letter_buffer:
            self.completed_words.append(''.join(self.letter_buffer))
            self.letter_buffer = []
            self.current_word = ""
            self.last_letter = None

    def get_caption(self):
        completed = ' '.join(self.completed_words)
        current = ''.join(self.letter_buffer)
        if completed and current:
            return f"{completed} {current}_"
        if completed:
            return completed
        if current:
            return f"{current}_"
        return ""

    def clear_caption(self):
        self.letter_buffer = []
        self.current_word = ""
        self.completed_words = []
        self.last_letter = None

    def add_space(self):
        self._finalize_word()

    def backspace(self):
        if self.letter_buffer:
            self.letter_buffer.pop()
            self.current_word = ''.join(self.letter_buffer)
        elif self.completed_words:
            last = self.completed_words.pop()
            self.letter_buffer = list(last)
            self.current_word = last

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self):
        return {
            'is_trained': self.is_trained,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'model_type': 'MLPClassifier' if self.is_trained else 'rule-based only',
        }

