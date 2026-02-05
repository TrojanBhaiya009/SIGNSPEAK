"""
Data Collection Script for Sign Language Recognition
Collects video sequences for training the ML model
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from sign_model import SignLanguageModel

class DataCollector:
    def __init__(self):
        self.model = SignLanguageModel()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        self.sequences = []
        self.labels = []
        self.current_word = None
        self.collecting = False
        self.frame_count = 0
        self.max_frames = 30
        
        # Create data directory
        Path('training_data').mkdir(exist_ok=True)
    
    def collect_data(self):
        """Main data collection loop"""
        print("\n🤟 Sign Language Data Collector")
        print("=" * 50)
        print("Controls:")
        print("  'c' - Start/stop collection")
        print("  'r' - Record a complete sequence")
        print("  's' - Save collected data")
        print("  'q' - Quit\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Process hand landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    for lm in landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                
                # If collecting, extract landmarks
                if self.collecting:
                    landmarks_data = self.model.extract_landmarks(frame)
                    self.sequences.append(landmarks_data)
                    self.frame_count += 1
                    
                    # Draw status
                    cv2.putText(frame, f"Recording: {self.frame_count}/{self.max_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display info
            if self.current_word:
                cv2.putText(frame, f"Word: {self.current_word}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.putText(frame, f"Sequences: {len(self.sequences)}", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Data Collection - Press Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.toggle_collection()
            elif key == ord('r'):
                self.record_sequence()
            elif key == ord('s'):
                self.save_data()
            elif key == ord('w'):
                self.current_word = input("\nEnter word name: ").upper()
    
    def toggle_collection(self):
        """Toggle data collection"""
        if not self.current_word:
            print("❌ Please set word name first (press 'w')")
            return
        
        self.collecting = not self.collecting
        self.frame_count = 0
        status = "Started" if self.collecting else "Stopped"
        print(f"✓ Collection {status} for '{self.current_word}'")
    
    def record_sequence(self):
        """Save current sequence"""
        if len(self.sequences) < self.max_frames:
            print(f"❌ Need {self.max_frames} frames, have {len(self.sequences)}")
            return
        
        # Get last 30 frames
        sequence = np.array(self.sequences[-self.max_frames:])
        self.labels.append(self.current_word)
        self.sequences = self.sequences[:-self.max_frames]
        
        print(f"✓ Recorded sequence for '{self.current_word}'")
        print(f"  Total sequences: {len(self.labels)}")
        
        self.collecting = False
        self.frame_count = 0
    
    def save_data(self):
        """Save training data"""
        if len(self.labels) == 0:
            print("❌ No data to save!")
            return
        
        # Prepare data
        X_train = []
        y_train = []
        
        # Collect all complete sequences
        while len(self.sequences) >= self.max_frames:
            sequence = np.array(self.sequences[:self.max_frames])
            X_train.append(sequence)
            y_train.append(self.current_word)
            self.sequences = self.sequences[self.max_frames:]
        
        # Add recorded sequences
        for _ in self.labels:
            if X_train:
                X_train.append(X_train[-1])  # Placeholder
                y_train.append(self.labels[-1])
        
        # Save to files
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        np.save('training_data/X_train.npy', X_train)
        np.save('training_data/y_train.npy', y_train)
        
        print(f"\n✓ Data saved!")
        print(f"  Sequences: {len(X_train)}")
        print(f"  Shape: {X_train.shape}")
        print(f"  File: training_data/X_train.npy")
        
        # Train model
        choice = input("\nTrain model now? (y/n): ").lower()
        if choice == 'y':
            print("Training... this may take a while")
            self.model.train_on_data(X_train, y_train, epochs=20)
            print("✓ Model trained!")
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    collector = DataCollector()
    try:
        collector.collect_data()
    finally:
        collector.cleanup()
