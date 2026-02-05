"""
ASL Alphabet Data Collection & Training Script
Collects hand landmark data for each letter and trains an ML model

Uses ACTUAL ASL Fingerspelling signs - reference images shown on screen!
"""

import cv2
import numpy as np
import os
import sys
import pickle
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Please install mediapipe: pip install mediapipe")
    sys.exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ASL Fingerspelling descriptions for each letter
ASL_DESCRIPTIONS = {
    'A': "Fist with thumb on the side",
    'B': "Flat hand, fingers together, thumb tucked",
    'C': "Curved hand like holding a cup",
    'D': "Index up, other fingers touch thumb",
    'E': "Fingers curled, thumb tucked under",
    'F': "OK sign - index & thumb circle, others up",
    'G': "Index & thumb pointing sideways",
    'H': "Index & middle pointing sideways",
    'I': "Pinky up, fist closed",
    'J': "Pinky up, draw J in air",
    'K': "Index & middle up in V, thumb between",
    'L': "L shape - index up, thumb out",
    'M': "Thumb under 3 fingers",
    'N': "Thumb under 2 fingers",
    'O': "Fingers curved to touch thumb - O shape",
    'P': "Like K but pointing down",
    'Q': "Like G but pointing down",
    'R': "Cross index & middle fingers",
    'S': "Fist with thumb over fingers",
    'T': "Thumb between index & middle",
    'U': "Index & middle up together",
    'V': "Peace sign - index & middle in V",
    'W': "Index, middle, ring up spread",
    'X': "Index finger bent like hook",
    'Y': "Thumb & pinky out (hang loose)",
    'Z': "Index draws Z in air"
}

class ASLDataCollector:
    def __init__(self):
        self.alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.data_dir = Path('training_data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe
        self._init_hand_detector()
        
        # Training data
        self.X = []  # Features (hand landmarks)
        self.y = []  # Labels (letters)
        
        # Collection settings
        self.samples_per_letter = 50
        self.current_letter_idx = 0
        self.samples_collected = 0
        self.collecting = False
        
    def _init_hand_detector(self):
        """Initialize MediaPipe gesture recognizer"""
        model_path = os.path.join('backend', 'gesture_recognizer.task')
        
        if not os.path.exists(model_path):
            print("⏳ Downloading gesture recognizer model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
            os.makedirs('backend', exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
            print("✓ Model downloaded")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        print("✓ Hand detector initialized")
    
    def extract_features(self, frame):
        """Extract hand landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        
        results = self.recognizer.recognize(mp_image)
        
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            landmarks = []
            for landmark in results.hand_landmarks[0]:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), results.hand_landmarks[0]
        return None, None
    
    def draw_hand(self, frame, landmarks, h, w):
        """Draw hand landmarks on frame"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        points = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (0, 200, 0), 2)
    
    def collect_data(self):
        """Interactive data collection loop"""
        print("\n" + "="*60)
        print("🤟 ASL FINGERSPELLING DATA COLLECTOR")
        print("="*60)
        print("\n⚠️  USE ACTUAL ASL SIGNS - NOT random hand shapes!")
        print("    Search 'ASL fingerspelling alphabet' for reference")
        print("\nInstructions:")
        print("  1. Look at the ASL description on screen")
        print("  2. Form the CORRECT ASL fingerspelling sign")
        print("  3. Press SPACE to start/stop collecting")
        print("  4. Press N for next letter / P for previous")
        print("  5. Press S to save and train model")
        print("  6. Press Q to quit")
        print("\nCollecting", self.samples_per_letter, "samples per letter")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            h, w, _ = frame.shape
            
            # Current letter
            current_letter = self.alphabet[self.current_letter_idx]
            
            # Extract features
            features, landmarks = self.extract_features(frame)
            
            # Draw hand if detected
            if landmarks:
                self.draw_hand(frame, landmarks, h, w)
                
                # Collect sample if in collection mode
                if self.collecting and features is not None:
                    self.X.append(features)
                    self.y.append(current_letter)
                    self.samples_collected += 1
                    
                    if self.samples_collected >= self.samples_per_letter:
                        self.collecting = False
                        print(f"✓ Collected {self.samples_per_letter} samples for '{current_letter}'")
                        self.samples_collected = 0
                        # Auto-advance to next letter
                        if self.current_letter_idx < len(self.alphabet) - 1:
                            self.current_letter_idx += 1
            
            # Draw UI
            # Large letter display with ASL description
            cv2.rectangle(frame, (0, 0), (w, 160), (20, 20, 20), -1)
            cv2.putText(frame, current_letter, (30, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 6)
            
            # ASL description
            desc = ASL_DESCRIPTIONS.get(current_letter, "")
            cv2.putText(frame, f"ASL: {desc}", (120, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Show the ACTUAL ASL sign!", (120, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(frame, "Search 'ASL fingerspelling' for reference", (120, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            
            # Status
            status = "COLLECTING..." if self.collecting else "Ready (Press SPACE)"
            status_color = (0, 255, 0) if self.collecting else (255, 255, 0)
            cv2.putText(frame, status, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Progress for current letter
            progress = f"Letter {self.current_letter_idx + 1}/{len(self.alphabet)}"
            cv2.putText(frame, progress, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Samples collected
            if self.collecting:
                sample_text = f"Samples: {self.samples_collected}/{self.samples_per_letter}"
                cv2.putText(frame, sample_text, (w - 250, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Total data
            total_text = f"Total: {len(self.X)} samples"
            cv2.putText(frame, total_text, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Hand detection status
            hand_status = "Hand: Detected" if landmarks else "Hand: Not detected"
            hand_color = (0, 255, 0) if landmarks else (0, 0, 255)
            cv2.putText(frame, hand_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
            
            cv2.imshow('ASL Data Collector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - toggle collection
                if not self.collecting:
                    self.collecting = True
                    self.samples_collected = 0
                    print(f"▶ Started collecting for '{current_letter}'")
                else:
                    self.collecting = False
                    print(f"⏸ Paused collection")
            elif key == ord('n'):  # Next letter
                self.collecting = False
                self.samples_collected = 0
                self.current_letter_idx = min(self.current_letter_idx + 1, len(self.alphabet) - 1)
            elif key == ord('p'):  # Previous letter
                self.collecting = False
                self.samples_collected = 0
                self.current_letter_idx = max(self.current_letter_idx - 1, 0)
            elif key == ord('s'):  # Save and train
                self.save_and_train()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_and_train(self):
        """Save collected data and train the model"""
        if len(self.X) < 10:
            print("❌ Not enough data to train. Collect more samples.")
            return
        
        print(f"\n📊 Training model with {len(self.X)} samples...")
        
        X = np.array(self.X)
        y = np.array(self.y)
        
        # Save raw data
        np.save(self.data_dir / 'X_train.npy', X)
        np.save(self.data_dir / 'y_train.npy', y)
        print(f"✓ Data saved to {self.data_dir}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            verbose=True
        )
        
        print("⏳ Training neural network...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        
        print(f"\n✅ Training complete!")
        print(f"   Train accuracy: {train_acc*100:.1f}%")
        print(f"   Test accuracy:  {test_acc*100:.1f}%")
        
        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'classes': list(set(y))
        }
        
        model_path = os.path.join('backend', 'sign_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {model_path}")
        
        # Print class distribution
        print("\n📊 Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for letter, count in zip(unique, counts):
            print(f"   {letter}: {count} samples")

def main():
    collector = ASLDataCollector()
    collector.collect_data()

if __name__ == '__main__':
    main()
