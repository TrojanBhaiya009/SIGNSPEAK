"""
collect_digits.py
=================
Collect ASL digit (0-9) landmark data using your webcam.
Saves 63-dim MediaPipe landmark vectors directly to:
  training_data/x_digits.npy   (N, 63)
  training_data/y_digits.npy   (N,)

After collecting, run:
  python train_combined.py --digit-x training_data\\x_digits.npy
                           --digit-y training_data\\y_digits.npy

Controls:
  0-9  → select digit to record
  SPACE → start / stop recording
  S    → save all collected data
  Q    → quit
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ── MediaPipe ─────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
except ImportError:
    print("[ERROR] mediapipe not installed.  pip install mediapipe")
    raise

SAVE_DIR      = Path('training_data')
OUT_X         = SAVE_DIR / 'x_digits.npy'
OUT_Y         = SAVE_DIR / 'y_digits.npy'
TARGET        = 150    # samples per digit (change if you want more)
DIGITS        = [str(d) for d in range(10)]


# ── Normalisation — MUST match sign_model.py ─────────────────────────────────
def normalize_landmarks(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float64)   # (21, 3)
    pts -= pts[0].copy()
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-6:
        pts /= max_dist
    return pts.flatten()   # (63,)


# ── Main collector ────────────────────────────────────────────────────────────
def main():
    SAVE_DIR.mkdir(exist_ok=True)

    # Load existing data so we can append
    data: dict[str, list] = defaultdict(list)
    if OUT_X.exists() and OUT_Y.exists():
        X_old = np.load(OUT_X)
        y_old = np.load(OUT_Y).astype(str)
        for feat, label in zip(X_old, y_old):
            data[label].append(feat)
        print(f"[DATA] Loaded existing: {dict(Counter(y_old))}")
    else:
        print("[DATA] Starting fresh — no existing digit data")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    current_digit = '0'
    recording     = False
    frame_count   = 0

    BG     = (13,  17,  23)
    GREEN  = (0,   255, 150)
    YELLOW = (0,   215, 255)
    RED    = (70,  70,  255)
    WHITE  = (230, 230, 230)
    BLUE   = (255, 165,  70)

    print("\n🤟 ASL Digit Collector  (0-9)")
    print("  Press 0-9 to select digit")
    print("  SPACE = start/stop recording")
    print("  S = save,  Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        features = None

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                if features is None:
                    features = normalize_landmarks(hand_lm)

        # ── Record ────────────────────────────────────────────────────────────
        if recording and features is not None:
            data[current_digit].append(features)
            frame_count += 1
            if frame_count >= TARGET:
                recording   = False
                frame_count = 0
                print(f"  ✅ '{current_digit}' done — total: {len(data[current_digit])}")

        # ── HUD ───────────────────────────────────────────────────────────────
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 25, 35), -1)

        status_col = GREEN if recording else (RED if features is None else WHITE)
        status_txt = (f"RECORDING '{current_digit}'  [{frame_count}/{TARGET}]"
                      if recording
                      else f"Ready — digit: '{current_digit}'  (SPACE=record)")
        cv2.putText(frame, status_txt, (12, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, status_col, 2)

        # Progress bar while recording
        if recording:
            bar_w = int(frame_count / TARGET * (w - 20))
            cv2.rectangle(frame, (10, 55), (10 + bar_w, 62), GREEN, -1)
            cv2.rectangle(frame, (10, 55), (w - 10,     62), WHITE, 1)

        # No-hand warning
        if features is None:
            cv2.putText(frame, "No hand detected", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        # Per-digit counts (right panel)
        panel_x = w - 160
        cv2.rectangle(frame, (panel_x - 8, 70), (w - 4, 70 + 11 * 26), (20, 25, 35), -1)
        cv2.putText(frame, "Collected:", (panel_x, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        for i, d in enumerate(DIGITS):
            cnt   = len(data.get(d, []))
            col   = GREEN if cnt >= TARGET else (YELLOW if cnt > 0 else (150, 150, 150))
            bar   = '█' * int(cnt / TARGET * 10) if cnt else ''
            label = f"{d}: {cnt:>3}  {bar}"
            cv2.putText(frame, label, (panel_x, 115 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

        # Digit selector strip (bottom)
        strip_y = h - 50
        cv2.rectangle(frame, (0, strip_y - 10), (w, h), (20, 25, 35), -1)
        slot_w = w // 10
        for i, d in enumerate(DIGITS):
            x      = i * slot_w
            cnt    = len(data.get(d, []))
            bg_col = (0, 80, 0) if d == current_digit else (30, 30, 45)
            done   = cnt >= TARGET
            cv2.rectangle(frame, (x + 2, strip_y - 5), (x + slot_w - 2, h - 5), bg_col, -1)
            txt_col = GREEN if done else (YELLOW if cnt > 0 else WHITE)
            cv2.putText(frame, d, (x + slot_w // 2 - 8, strip_y + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        GREEN if d == current_digit else txt_col, 2)
            if done:
                cv2.putText(frame, '✓', (x + slot_w // 2 - 6, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)

        cv2.imshow('ASL Digit Collector  [0-9 = select | SPACE = record | S = save | Q = quit]', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord(d) for d in DIGITS]:
            current_digit = chr(key)
            recording     = False
            frame_count   = 0
            print(f"  Selected: '{current_digit}'  (have {len(data.get(current_digit, []))} samples)")
        elif key == ord(' '):
            if features is None:
                print("  ⚠ No hand in frame — show your hand first!")
            else:
                recording   = not recording
                frame_count = 0   # reset counter for new recording session
                if recording:
                    print(f"  ⏺ Recording '{current_digit}'…")
                else:
                    print(f"  ⏹ Stopped  '{current_digit}' — {len(data.get(current_digit,[]))} total")
        elif key == ord('s'):
            _save(data)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Auto-save on quit
    if any(data.values()):
        ans = input("\nSave before quitting? (y/n): ").strip().lower()
        if ans == 'y':
            _save(data)


def _save(data: dict):
    if not any(data.values()):
        print("  [SAVE] Nothing to save yet.")
        return

    X_list, y_list = [], []
    for label, vecs in sorted(data.items(), key=lambda x: int(x[0])):
        X_list.extend(vecs)
        y_list.extend([label] * len(vecs))

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)

    print(f"\n  [SAVE] ✅ Saved {len(X)} samples → {OUT_X}")
    print(f"         Classes: {dict(Counter(y))}")
    total_done = sum(1 for d in range(10) if len(data.get(str(d), [])) >= 150)
    print(f"         Digits complete (≥150 samples): {total_done}/10")

    if total_done == 10:
        print(f"""
  🚀 All 10 digits collected! Now train:
     python train_combined.py \\
         --digit-x training_data\\x_digits.npy \\
         --digit-y training_data\\y_digits.npy
        """)


if __name__ == '__main__':
    main()