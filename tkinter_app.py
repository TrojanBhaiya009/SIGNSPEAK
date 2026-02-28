"""
tkinter_app.py
==============
Hand Sign Language Trainer — standalone desktop demo.
No pre-trained model needed: record gestures in-app and train a KNN.

Supports:
  • Letters A-Z
  • Digits  0-9  (NEW)
  • Custom phrase labels

"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import json
import os
from datetime import datetime
from collections import deque, Counter

# Fix Keras 3.x / mediapipe compatibility — must be set before import
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# ── MediaPipe (Tasks API — 0.10.30+) ──────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"MediaPipe not found: {e}")

# Hand connections for manual drawing (replaces mp.solutions.drawing_utils)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand_landmarks(frame, landmarks):
    """Draw hand landmarks and connections on frame (replaces mp_drawing)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
    for px, py in pts:
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX-2 + FIX-3: Canonical 63-dim feature extraction
#  (matches sign_model.py / collect_data.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(hand_landmarks) -> np.ndarray:
    """
    Extract a 63-dim feature vector from a MediaPipe hand_landmarks object.

    Normalisation:
      1. Centre on wrist (landmark 0)         — position invariant
      2. Scale by max landmark distance        — scale invariant

    FIX-2: now uses x,y,z (63-dim) instead of x,y only (42-dim).
    FIX-3: scale denominator changed from lm-9 distance to max distance,
           matching sign_model.py normalize_landmarks().
    """
    # Works with both old (hand_landmarks.landmark) and new Tasks API (list)
    lms = hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks.landmark
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                   dtype=np.float32)   # (21, 3)

    # Centre on wrist
    pts -= pts[0].copy()

    # Scale by maximum distance from wrist (FIX-3)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-6:
        pts /= max_dist

    return pts.flatten()   # (63,)


# ═══════════════════════════════════════════════════════════════════════════════
#  KNN Classifier — pure numpy, no sklearn needed
# ═══════════════════════════════════════════════════════════════════════════════

class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k        = k
        self.X        = np.empty((0, 63), dtype=np.float32)
        self.y: list  = []
        self.trained  = False

    def fit(self, X: np.ndarray, y: list):
        self.X       = np.array(X, dtype=np.float32)
        self.y       = list(y)
        self.trained = len(X) > 0

    def predict(self, x: np.ndarray) -> tuple:
        """Return (label, confidence 0-1).  O(n) — fine for demo scale."""
        if not self.trained or len(self.X) == 0:
            return None, 0.0
        x     = np.array(x, dtype=np.float32)
        dists = np.linalg.norm(self.X - x, axis=1)
        k_idx = np.argsort(dists)[: min(self.k, len(self.X))]
        votes = Counter(self.y[i] for i in k_idx)
        best_label, best_votes = votes.most_common(1)[0]
        vote_conf  = best_votes / len(k_idx)
        dist_conf  = 1.0 / (1.0 + dists[k_idx[0]] * 5)
        confidence = round(vote_conf * 0.6 + dist_conf * 0.4, 3)
        return best_label, confidence

    def n_samples(self) -> int:
        return len(self.y)

    def label_counts(self) -> dict:
        return dict(Counter(self.y))

    # NEW: simple leave-one-out accuracy estimate
    def loo_accuracy(self, max_samples: int = 500) -> float:
        """
        Approximate Leave-One-Out accuracy on up to max_samples samples.
        Gives the user feedback on model quality after training.
        """
        if not self.trained or self.n_samples() < 2:
            return 0.0
        n      = min(self.n_samples(), max_samples)
        idx    = np.random.choice(self.n_samples(), n, replace=False)
        X_sub  = self.X[idx]
        y_sub  = [self.y[i] for i in idx]
        correct = 0
        for i in range(n):
            # Build a temporary predictor excluding sample i
            mask = np.ones(n, dtype=bool); mask[i] = False
            X_tr = X_sub[mask]
            y_tr = [y_sub[j] for j in range(n) if j != i]
            if len(set(y_tr)) < 2:
                continue
            dists = np.linalg.norm(X_tr - X_sub[i], axis=1)
            k_idx = np.argsort(dists)[: min(self.k, len(X_tr))]
            votes = Counter(y_tr[j] for j in k_idx)
            if votes.most_common(1)[0][0] == y_sub[i]:
                correct += 1
        return correct / n


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class HandSignTrainerApp:
    # Palette
    BG     = "#0d1117"; PANEL  = "#161b22"; CARD   = "#1f2937"
    GREEN  = "#00ff96"; RED    = "#ff4d6d"; BLUE   = "#58a6ff"
    YELLOW = "#ffd700"; PURPLE = "#a78bfa"; TEXT   = "#e6edf3"
    DIM    = "#8b949e"; BORDER = "#30363d"; SKY    = "#38bdf8"

    # NEW: all valid labels
    ALL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ALL_DIGITS  = [str(d) for d in range(10)]
    ALL_LABELS  = ALL_LETTERS + ALL_DIGITS

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🤟 Hand Sign Trainer — A-Z + 0-9")
        self.root.geometry("1540x940")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)

        # Camera
        self.camera_active = False
        self.cap           = None
        self.hands_mp      = None

        # Classifier
        self.knn           = KNNClassifier(k=5)
        self.training_data: dict[str, list] = {}
        self.model_trained = False

        # Smoothing
        self.pred_history    = deque(maxlen=10)
        self.last_prediction = "—"
        self.last_conf       = 0.0

        # Recording
        self.recording       = False
        self.record_label    = ""
        self.record_samples: list = []
        self.record_target   = 60

        # Transcript
        self.transcript   = ""
        self.last_added   = ""
        # FIX-1: single cooldown counter
        self.add_cooldown = 0   # decremented ONCE per frame (was twice — bug)

        if MEDIAPIPE_AVAILABLE:
            # Locate hand_landmarker.task model file
            _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'backend', 'hand_landmarker.task')
            if not os.path.exists(_model_path):
                _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'hand_landmarker.task')
            base_opts = mp_python.BaseOptions(model_asset_path=_model_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=1,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.hands_mp = mp_vision.HandLandmarker.create_from_options(opts)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-start camera after UI is ready
        if MEDIAPIPE_AVAILABLE:
            self.root.after(500, self._start_camera)

    # ─────────────────────────────────────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        bar = tk.Frame(self.root, bg="#0a0e14", height=52)
        bar.pack(side=tk.TOP, fill=tk.X); bar.pack_propagate(False)
        tk.Label(bar, text="🤟  Hand Sign Language Trainer  A-Z + 0-9",
                 font=("Arial", 17, "bold"), fg=self.GREEN, bg="#0a0e14"
                 ).pack(side=tk.LEFT, padx=18, pady=10)
        self.status_bar = tk.Label(bar, text="Camera OFF",
                                   font=("Arial", 11), fg=self.DIM, bg="#0a0e14")
        self.status_bar.pack(side=tk.RIGHT, padx=18)

        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Video (left)
        left = tk.Frame(main, bg=self.BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH)
        self.video_label = tk.Label(left, bg="#000", width=700, height=520)
        self.video_label.pack(padx=4, pady=4)

        pred_card = tk.Frame(left, bg=self.CARD)
        pred_card.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(pred_card, text="Detected Sign:", font=("Arial", 11),
                 fg=self.DIM, bg=self.CARD).pack(side=tk.LEFT, padx=10, pady=10)
        self.pred_label = tk.Label(pred_card, text="—", font=("Arial", 24, "bold"),
                                   fg=self.GREEN, bg=self.CARD)
        self.pred_label.pack(side=tk.LEFT, padx=8)
        self.conf_label = tk.Label(pred_card, text="conf: 0%",
                                   font=("Arial", 11), fg=self.DIM, bg=self.CARD)
        self.conf_label.pack(side=tk.RIGHT, padx=14)
        self.conf_bar_var = tk.DoubleVar()
        ttk.Progressbar(pred_card, variable=self.conf_bar_var,
                        maximum=100, length=200, mode='determinate'
                        ).pack(side=tk.RIGHT, padx=6)

        # Notebook (right)
        right = tk.Frame(main, bg=self.BG, width=460)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        right.pack_propagate(False)

        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)
        t1 = tk.Frame(nb, bg=self.PANEL)
        t2 = tk.Frame(nb, bg=self.PANEL)
        t3 = tk.Frame(nb, bg=self.PANEL)
        nb.add(t1, text="  🏋 Train  ")
        nb.add(t2, text="  📊 Model  ")
        nb.add(t3, text="  📝 Transcript  ")
        self._build_train_tab(t1)
        self._build_model_tab(t2)
        self._build_transcript_tab(t3)

        btn_row = tk.Frame(right, bg=self.BG)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        self.cam_btn = self._btn(btn_row, "▶  START CAMERA", self._start_camera,
                                 self.GREEN, "#000")
        self.cam_btn.pack(side=tk.LEFT, padx=4, pady=4, expand=True, fill=tk.X)
        self.stop_btn = self._btn(btn_row, "⏹  STOP", self._stop_camera,
                                  self.RED, "#fff", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4, pady=4, expand=True, fill=tk.X)

    def _btn(self, parent, text, cmd, bg, fg, state=tk.NORMAL, **kw):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Arial", 11, "bold"), bg=bg, fg=fg,
                         relief=tk.FLAT, activebackground=bg,
                         cursor="hand2", state=state, **kw)

    # ── Train tab ─────────────────────────────────────────────────────────────
    def _build_train_tab(self, parent):
        tk.Label(parent, text="MANUAL GESTURE TRAINING",
                 font=("Arial", 13, "bold"), fg=self.GREEN,
                 bg=self.PANEL).pack(pady=(16, 4))
        tk.Label(parent,
                 text="1. Type a label (A-Z or 0-9)  2. Hit Record  3. Show gesture",
                 font=("Arial", 9), fg=self.DIM, bg=self.PANEL).pack(pady=2)

        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill=tk.X, padx=14, pady=8)
        tk.Label(row, text="Sign Label:", font=("Arial", 11),
                 fg=self.TEXT, bg=self.PANEL).pack(side=tk.LEFT)
        self.label_var = tk.StringVar()
        tk.Entry(row, textvariable=self.label_var, width=12,
                 font=("Arial", 13, "bold"), bg=self.CARD,
                 fg=self.YELLOW, insertbackground=self.TEXT,
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=8, ipady=4)

        # NEW: quick-select buttons for digits
        dig_row = tk.Frame(parent, bg=self.PANEL)
        dig_row.pack(fill=tk.X, padx=14, pady=(0, 4))
        tk.Label(dig_row, text="Quick:", font=("Arial", 9),
                 fg=self.DIM, bg=self.PANEL).pack(side=tk.LEFT)
        for d in self.ALL_DIGITS:
            tk.Button(dig_row, text=d, width=2,
                      font=("Arial", 9, "bold"),
                      bg=self.CARD, fg=self.SKY, relief=tk.FLAT,
                      cursor="hand2",
                      command=lambda lbl=d: self.label_var.set(lbl)
                      ).pack(side=tk.LEFT, padx=1)

        row2 = tk.Frame(parent, bg=self.PANEL)
        row2.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(row2, text="Samples to record:", font=("Arial", 10),
                 fg=self.DIM, bg=self.PANEL).pack(side=tk.LEFT)
        self.n_samples_var = tk.IntVar(value=60)
        tk.Spinbox(row2, from_=20, to=300, textvariable=self.n_samples_var,
                   width=6, font=("Arial", 11), bg=self.CARD, fg=self.TEXT,
                   buttonbackground=self.CARD).pack(side=tk.LEFT, padx=8)

        self.record_btn = self._btn(parent, "⏺  RECORD GESTURE",
                                    self._start_recording, self.BLUE, "#fff")
        self.record_btn.pack(fill=tk.X, padx=14, pady=6, ipady=6)

        self.rec_status = tk.Label(parent, text="",
                                   font=("Arial", 11, "bold"),
                                   fg=self.YELLOW, bg=self.PANEL)
        self.rec_status.pack(pady=2)
        self.rec_prog_var = tk.DoubleVar()
        ttk.Progressbar(parent, variable=self.rec_prog_var,
                        maximum=100, length=360).pack(pady=4)

        self.train_btn = self._btn(parent, "🚀  TRAIN MODEL",
                                   self._train_model, self.PURPLE, "#fff")
        self.train_btn.pack(fill=tk.X, padx=14, pady=(14, 4), ipady=8)

        self.train_result = tk.Label(parent, text="No model trained yet.",
                                     font=("Arial", 10), fg=self.DIM,
                                     bg=self.PANEL, wraplength=400)
        self.train_result.pack(pady=4)

        tk.Label(parent, text="Recorded Gestures",
                 font=("Arial", 11, "bold"), fg=self.DIM,
                 bg=self.PANEL).pack(pady=(12, 2))
        self.gestures_box = tk.Text(parent, height=7, bg=self.CARD,
                                    fg=self.TEXT, font=("Courier", 10),
                                    relief=tk.FLAT, state=tk.DISABLED)
        self.gestures_box.pack(fill=tk.X, padx=14, pady=4)

        br = tk.Frame(parent, bg=self.PANEL); br.pack(fill=tk.X, padx=14)
        self._btn(br, "🗑 Clear All",  self._clear_all_data, "#333", self.RED ).pack(side=tk.LEFT, padx=2)
        self._btn(br, "💾 Save Data",  self._save_data,      "#333", self.GREEN).pack(side=tk.LEFT, padx=2)
        self._btn(br, "📂 Load Data",  self._load_data,      "#333", self.BLUE ).pack(side=tk.LEFT, padx=2)
        self._btn(br, "📤 Export NPY", self._export_npy,     "#333", self.YELLOW).pack(side=tk.LEFT, padx=2)

    # ── Model tab ─────────────────────────────────────────────────────────────
    def _build_model_tab(self, parent):
        tk.Label(parent, text="MODEL INFO", font=("Arial", 13, "bold"),
                 fg=self.GREEN, bg=self.PANEL).pack(pady=(16, 4))
        self.model_info = tk.Text(parent, height=24, bg=self.CARD, fg=self.TEXT,
                                  font=("Courier", 10), relief=tk.FLAT,
                                  state=tk.DISABLED)
        self.model_info.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)
        row = tk.Frame(parent, bg=self.PANEL); row.pack(fill=tk.X, padx=14, pady=6)
        tk.Label(row, text="KNN k:", font=("Arial", 11),
                 fg=self.TEXT, bg=self.PANEL).pack(side=tk.LEFT)
        self.k_var = tk.IntVar(value=5)
        tk.Spinbox(row, from_=1, to=21, textvariable=self.k_var,
                   width=5, font=("Arial", 11), bg=self.CARD, fg=self.TEXT,
                   command=self._update_k).pack(side=tk.LEFT, padx=8)
        self._btn(parent, "🔄 Refresh", self._refresh_model_info,
                  "#333", self.BLUE).pack(pady=4)

    # ── Transcript tab ────────────────────────────────────────────────────────
    def _build_transcript_tab(self, parent):
        tk.Label(parent, text="LIVE TRANSCRIPT", font=("Arial", 13, "bold"),
                 fg=self.GREEN, bg=self.PANEL).pack(pady=(16, 4))
        self.transcript_text = tk.Text(parent, bg=self.CARD, fg=self.GREEN,
                                       font=("Courier", 13), relief=tk.FLAT,
                                       wrap=tk.WORD)
        self.transcript_text.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)

        thr = tk.Frame(parent, bg=self.PANEL); thr.pack(fill=tk.X, padx=14, pady=4)
        tk.Label(thr, text="Min confidence:", font=("Arial", 9),
                 fg=self.DIM, bg=self.PANEL).pack(side=tk.LEFT)
        self.min_conf_var = tk.DoubleVar(value=0.7)
        tk.Scale(thr, from_=0.3, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.min_conf_var, bg=self.PANEL, fg=self.TEXT,
                 troughcolor=self.CARD, length=150).pack(side=tk.LEFT)

        br = tk.Frame(parent, bg=self.PANEL); br.pack(fill=tk.X, padx=14, pady=4)
        self._btn(br, "🗑 Clear",  self._clear_transcript, "#333", self.RED  ).pack(side=tk.LEFT, padx=4)
        self._btn(br, "💾 Save",   self._save_transcript,  "#333", self.GREEN).pack(side=tk.LEFT, padx=4)

    # ─────────────────────────────────────────────────────────────────────────
    #  Camera
    # ─────────────────────────────────────────────────────────────────────────
    def _start_camera(self):
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe not installed.\npip install mediapipe")
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  700)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)
        self.camera_active = True
        self.cam_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_bar.config(text="🟢 Camera ACTIVE", fg=self.GREEN)
        threading.Thread(target=self._video_loop, daemon=True).start()

    def _stop_camera(self):
        self.camera_active = False
        if self.cap: self.cap.release(); self.cap = None
        self.cam_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Camera OFF", fg=self.DIM)

    # ─────────────────────────────────────────────────────────────────────────
    #  Video loop
    # ─────────────────────────────────────────────────────────────────────────
    def _video_loop(self):
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret: continue

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = self.hands_mp.detect(mp_image)
            features = None

            if results.hand_landmarks:
                for hand_lm in results.hand_landmarks:
                    draw_hand_landmarks(frame, hand_lm)
                    if features is None:
                        # FIX-2 + FIX-3: use the new 63-dim extractor
                        features = extract_features(hand_lm)

            # ── Record ────────────────────────────────────────────────────
            if self.recording and features is not None:
                self.record_samples.append(features)
                done = len(self.record_samples)
                pct  = done / self.record_target * 100
                self.root.after(0, self._update_record_progress, done, pct)
                if done >= self.record_target:
                    self.root.after(0, self._finish_recording)

            # ── Predict ───────────────────────────────────────────────────
            if self.model_trained and features is not None and not self.recording:
                label, conf = self.knn.predict(features)
                self.pred_history.append((label, conf))
                labels    = [p[0] for p in self.pred_history]
                best      = Counter(labels).most_common(1)[0][0]
                avg_conf  = float(np.mean([p[1] for p in self.pred_history if p[0] == best]))
                self.last_prediction = best
                self.last_conf       = avg_conf

                # Transcript — add word if confident and different from last
                if conf >= self.min_conf_var.get() and best != self.last_added:
                    if self.add_cooldown <= 0:
                        self.transcript  += f" {best}"
                        self.last_added   = best
                        self.add_cooldown = 25
                        self.root.after(0, self._update_transcript_display)

                # FIX-1: decrement exactly ONCE per frame (was decremented twice)
                self.add_cooldown = max(0, self.add_cooldown - 1)

            elif not self.recording:
                self.last_prediction = "No hand" if features is None else "—"
                self.last_conf       = 0.0

            self._draw_overlay(frame, features)
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.root.after(0, self._set_frame, imgtk)
            self.root.after(0, self._update_pred_ui)

    def _draw_overlay(self, frame, features):
        h, w = frame.shape[:2]
        if self.recording:
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 80, 200), -1)
            cv2.putText(frame,
                        f"RECORDING: {self.record_label}  "
                        f"[{len(self.record_samples)}/{self.record_target}]",
                        (12, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        (255, 255, 0), 2)
        elif self.model_trained and features is not None:
            col = (0, 255, 150) if self.last_conf >= 0.7 else (200, 200, 0)
            cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
            cv2.putText(frame,
                        f"{self.last_prediction}  ({self.last_conf*100:.0f}%)",
                        (12, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, col, 2)
        elif not self.model_trained:
            cv2.putText(frame, "Train a model first →",
                        (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (100, 100, 200), 2)

    def _set_frame(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _update_pred_ui(self):
        self.pred_label.config(text=self.last_prediction)
        pct = int(self.last_conf * 100)
        self.conf_label.config(text=f"conf: {pct}%")
        self.conf_bar_var.set(pct)

    # ─────────────────────────────────────────────────────────────────────────
    #  Recording
    # ─────────────────────────────────────────────────────────────────────────
    def _start_recording(self):
        label = self.label_var.get().strip().upper()
        if not label:
            messagebox.showwarning("No Label", "Enter a sign label first.")
            return
        if not self.camera_active:
            messagebox.showwarning("Camera Off", "Start the camera first.")
            return
        self.record_label   = label
        self.record_target  = self.n_samples_var.get()
        self.record_samples = []
        self.recording      = True
        self.record_btn.config(state=tk.DISABLED)
        self.rec_status.config(text=f"⏺ Recording '{label}' …")

    def _update_record_progress(self, done, pct):
        self.rec_status.config(
            text=f"⏺ '{self.record_label}': {done}/{self.record_target}")
        self.rec_prog_var.set(pct)

    def _finish_recording(self):
        self.recording = False
        label = self.record_label
        self.training_data.setdefault(label, []).extend(self.record_samples)
        n = len(self.training_data[label])
        self.rec_status.config(text=f"✅ '{label}' saved! Total: {n}")
        self.rec_prog_var.set(100)
        self.record_btn.config(state=tk.NORMAL)
        self._refresh_gestures_list()

    # ─────────────────────────────────────────────────────────────────────────
    #  Training
    # ─────────────────────────────────────────────────────────────────────────
    def _train_model(self):
        if len(self.training_data) < 2:
            messagebox.showwarning("Not Enough Data",
                                   "Record at least 2 different gestures first.")
            return
        X, y = [], []
        for label, vecs in self.training_data.items():
            X.extend(vecs); y.extend([label] * len(vecs))
        self.knn.k = self.k_var.get()
        self.knn.fit(np.array(X), y)
        self.model_trained = True

        # NEW: compute leave-one-out accuracy for user feedback
        self.train_result.config(text="Computing accuracy …", fg=self.DIM)
        self.root.update()
        loo_acc = self.knn.loo_accuracy()

        n_cls, n_tot = len(self.training_data), len(X)
        msg = (f"✅ Model trained!\n"
               f"  {n_cls} gestures  |  {n_tot} samples\n"
               f"  KNN k={self.knn.k}  |  LOO accuracy: {loo_acc:.1%}")
        col = self.GREEN if loo_acc >= 0.85 else self.YELLOW if loo_acc >= 0.70 else self.RED
        self.train_result.config(text=msg, fg=col)
        self._refresh_model_info()
        self.pred_history.clear()

    def _update_k(self):
        self.knn.k = self.k_var.get()
        self._refresh_model_info()

    # ─────────────────────────────────────────────────────────────────────────
    #  Transcript
    # ─────────────────────────────────────────────────────────────────────────
    def _update_transcript_display(self):
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.delete("1.0", tk.END)
        self.transcript_text.insert("1.0", self.transcript.strip())
        self.transcript_text.see(tk.END)

    def _clear_transcript(self):
        self.transcript = ""; self.last_added = ""
        self._update_transcript_display()

    def _save_transcript(self):
        if not self.transcript.strip():
            messagebox.showwarning("Empty", "Nothing to save."); return
        fp = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if fp:
            with open(fp, "w") as f: f.write(self.transcript.strip())
            messagebox.showinfo("Saved", f"Saved to:\n{fp}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Model info
    # ─────────────────────────────────────────────────────────────────────────
    def _refresh_gestures_list(self):
        self.gestures_box.config(state=tk.NORMAL)
        self.gestures_box.delete("1.0", tk.END)
        if not self.training_data:
            self.gestures_box.insert("1.0", "(no data yet)")
        else:
            # Letters first, then digits, then other
            letters = sorted([k for k in self.training_data if k.isalpha()])
            digits  = sorted([k for k in self.training_data if k.isdigit()], key=int)
            others  = sorted([k for k in self.training_data if not k.isalpha() and not k.isdigit()])
            for label in letters + digits + others:
                vecs = self.training_data[label]
                self.gestures_box.insert(tk.END, f"  {label:14s} {len(vecs):>4} samples\n")
        self.gestures_box.config(state=tk.DISABLED)

    def _refresh_model_info(self):
        self.model_info.config(state=tk.NORMAL)
        self.model_info.delete("1.0", tk.END)
        lines = [
            "══ KNN Classifier ══",
            f"  Algorithm  : K-Nearest Neighbors",
            f"  k          : {self.knn.k}",
            f"  Features   : 63  (21 landmarks × x,y,z)",  # FIX-2
            f"  Norm       : Wrist-centred + max-scale",    # FIX-3
            "",
        ]
        if self.model_trained:
            lc    = self.knn.label_counts()
            total = self.knn.n_samples()
            letters_in = sorted([k for k in lc if k.isalpha()])
            digits_in  = sorted([k for k in lc if k.isdigit()], key=int)
            lines += [
                f"  Status     : ✅ TRAINED",
                f"  Classes    : {len(lc)}  "
                f"({len(letters_in)} letters + {len(digits_in)} digits)",
                f"  Total      : {total} samples",
                "",
                "  Per-class (letters):",
            ]
            max_c = max(lc.values())
            for label in letters_in:
                bar = "█" * int(lc[label] / max_c * 20)
                lines.append(f"    {label:5s} {lc[label]:>4}  {bar}")
            if digits_in:
                lines.append("")
                lines.append("  Per-class (digits):")
                for label in digits_in:
                    bar = "█" * int(lc[label] / max_c * 20)
                    lines.append(f"    {label:5s} {lc[label]:>4}  {bar}")
        else:
            lines += [
                "  Status     : ⚠ Not trained",
                "  → Record gestures and hit TRAIN MODEL",
            ]
        self.model_info.insert("1.0", "\n".join(lines))
        self.model_info.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────────────
    #  Save / Load
    # ─────────────────────────────────────────────────────────────────────────
    def _clear_all_data(self):
        if not messagebox.askyesno("Confirm", "Delete ALL recorded data?"): return
        self.training_data = {}; self.model_trained = False
        self.knn = KNNClassifier(k=self.k_var.get())
        self.train_result.config(text="All data cleared.", fg=self.RED)
        self._refresh_gestures_list(); self._refresh_model_info()

    def _save_data(self):
        if not self.training_data:
            messagebox.showwarning("No Data", "Nothing recorded yet."); return
        fp = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not fp: return
        data = {k: [v.tolist() for v in vecs] for k, vecs in self.training_data.items()}
        with open(fp, "w") as f: json.dump(data, f)
        messagebox.showinfo("Saved", f"Saved:\n{fp}")

    def _load_data(self):
        fp = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not fp: return
        with open(fp) as f: data = json.load(f)
        self.training_data = {k: [np.array(v, dtype=np.float32) for v in vecs]
                               for k, vecs in data.items()}
        n_total = sum(len(v) for v in self.training_data.values())
        messagebox.showinfo("Loaded",
                            f"Loaded {n_total} samples across "
                            f"{len(self.training_data)} gestures.")
        self._refresh_gestures_list()


    # ─────────────────────────────────────────────────────────
    #  Export to X_train.npy + y_train.npy
    # ─────────────────────────────────────────────────────────
    def _export_npy(self):
        """
        Export data as X_train.npy + y_train.npy into training_data/
        so train_alphabet.py / train_combined.py can use it directly.
        """
        if not self.training_data:
            messagebox.showwarning("No Data", "Nothing recorded yet.")
            return

        X_list, y_list = [], []
        for label, vecs in self.training_data.items():
            for v in vecs:
                arr = np.array(v, dtype=np.float64).flatten()
                if arr.shape[0] == 63:
                    X_list.append(arr)
                    y_list.append(label)

        if not X_list:
            messagebox.showerror("Error", "No valid 63-dim samples found.")
            return

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list)

        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
        os.makedirs(default_dir, exist_ok=True)

        x_path = os.path.join(default_dir, 'X_train.npy')
        y_path = os.path.join(default_dir, 'y_train.npy')

        if os.path.exists(x_path):
            if not messagebox.askyesno("Overwrite?",
                    f"X_train.npy already exists.\nOverwrite with {len(X)} samples?"):
                return

        np.save(x_path, X)
        np.save(y_path, y)

        from collections import Counter
        counts = dict(Counter(y_list))
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        messagebox.showinfo("Exported to NPY",
            f"Saved {len(X)} samples to training_data/\n\n"
            f"Classes: {summary}\n\n"
            f"Next step:\n"
            f"  python train_combined.py --digit-x training_data/x_digits.npy --digit-y training_data/y_digits.npy")

    # ─────────────────────────────────────────────────────────────────────────
    def _on_close(self):
        self.camera_active = False
        if self.cap: self.cap.release()
        if self.hands_mp: self.hands_mp.close()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = HandSignTrainerApp(root)
    root.mainloop()