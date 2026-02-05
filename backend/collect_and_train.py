"""
MANUAL TRAINING SCRIPT - Collect your own sign data and train
=============================================================

Run this script to:
1. Collect hand sign images for each letter using your webcam
2. Train a personalized model on YOUR hand signs

USAGE:
    python collect_and_train.py

CONTROLS:
    - Press the letter key (A-Z) to start collecting for that letter
    - Press 'S' to skip to training with current data
    - Press 'Q' to quit
    - Press 'T' to train with existing public folder data
    - Press 'R' to retrain with your new data only
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from datetime import datetime
import sys

# Add path for imports
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / 'my_training_data'  # Your personal training data (in D:\Hackathon)
PUBLIC_DIR = SCRIPT_DIR.parent / 'public'   # Existing ISL images
MODEL_PATH = SCRIPT_DIR / 'sign_model_cnn.pkl'
IMG_SIZE = 64
SAMPLES_PER_LETTER = 50  # Collect 50 images per letter

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def collect_data():
    """Collect training images from webcam"""
    print("\n" + "="*60)
    print("  MANUAL DATA COLLECTION")
    print("="*60)
    print("""
INSTRUCTIONS:
1. Show a sign and press the letter key (A-Z) to collect
2. Hold the sign steady - it will capture 50 images quickly
3. Move your hand slightly for variety
4. Press 'Q' to quit, 'T' to train with all data

TIP: Good lighting + plain background = better results!
    """)
    
    ensure_dir(DATA_DIR)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return False
    
    collected = {chr(i): 0 for i in range(65, 91)}  # A-Z
    current_letter = None
    collect_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror for natural feel
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Draw collection status
        y = 30
        for i, letter in enumerate('ABCDEFGHIJKLM'):
            color = (0, 255, 0) if collected[letter] >= SAMPLES_PER_LETTER else (0, 100, 255)
            cv2.putText(display, f"{letter}:{collected[letter]}", (10 + i*45, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y = 50
        for i, letter in enumerate('NOPQRSTUVWXYZ'):
            color = (0, 255, 0) if collected[letter] >= SAMPLES_PER_LETTER else (0, 100, 255)
            cv2.putText(display, f"{letter}:{collected[letter]}", (10 + i*45, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Show ROI (region of interest)
        h, w = frame.shape[:2]
        roi_size = 200
        x1, y1 = w//2 - roi_size//2, h//2 - roi_size//2
        x2, y2 = x1 + roi_size, y1 + roi_size
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(display, "Place hand here", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if current_letter:
            cv2.putText(display, f"Collecting '{current_letter}': {collect_count}/{SAMPLES_PER_LETTER}", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save ROI
            roi = frame[y1:y2, x1:x2]
            letter_dir = DATA_DIR / current_letter
            ensure_dir(letter_dir)
            
            timestamp = datetime.now().strftime("%H%M%S%f")
            cv2.imwrite(str(letter_dir / f"{current_letter}_{timestamp}.jpg"), roi)
            
            collect_count += 1
            collected[current_letter] += 1
            
            if collect_count >= SAMPLES_PER_LETTER:
                print(f"✓ Collected {SAMPLES_PER_LETTER} images for '{current_letter}'")
                current_letter = None
                collect_count = 0
        else:
            cv2.putText(display, "Press A-Z to collect, T to train, Q to quit", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Collect Training Data", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('t') or key == ord('T'):
            cap.release()
            cv2.destroyAllWindows()
            return True
        elif chr(key).upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            current_letter = chr(key).upper()
            collect_count = 0
            print(f"\n→ Collecting for '{current_letter}'... Hold your sign steady!")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def load_and_preprocess_image(image_path, size=64):
    """Load, convert to grayscale, resize, normalize, flatten"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size, size))
        normalized = resized.astype(np.float32) / 255.0
        return normalized.flatten()
    except:
        return None

def load_dataset(data_dirs, img_size=64, max_per_letter=200, min_per_letter=10):
    """Load images from multiple directories"""
    X = []
    y = []
    
    print("\n📊 Loading training data...")
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        print(f"\n  From: {data_dir}")
        
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            letter_dir = data_dir / letter
            if not letter_dir.exists():
                continue
            
            images = list(letter_dir.glob('*.jpg')) + list(letter_dir.glob('*.png'))
            if len(images) < min_per_letter:
                continue
            
            # Sample evenly
            if len(images) > max_per_letter:
                indices = np.linspace(0, len(images)-1, max_per_letter, dtype=int)
                images = [images[i] for i in indices]
            
            loaded = 0
            for img_path in images:
                features = load_and_preprocess_image(img_path, img_size)
                if features is not None:
                    X.append(features)
                    y.append(letter)
                    loaded += 1
            
            if loaded > 0:
                print(f"    {letter}: {loaded} images")
    
    return np.array(X), np.array(y)

def train_model(use_my_data=True, use_public_data=True):
    """Train the model"""
    print("\n" + "="*60)
    print("  TRAINING MODEL")
    print("="*60)
    
    # Collect data from specified sources
    data_dirs = []
    if use_my_data:
        data_dirs.append(DATA_DIR)
    if use_public_data:
        data_dirs.append(PUBLIC_DIR)
    
    if not data_dirs:
        print("ERROR: No data directories specified!")
        return
    
    X, y = load_dataset(data_dirs, img_size=IMG_SIZE, max_per_letter=300)
    
    if len(X) == 0:
        print("ERROR: No training data found!")
        return
    
    print(f"\n📈 Total: {len(X)} samples, {len(set(y))} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n🧠 Training neural network...")
    print("   (This may take 1-3 minutes)")
    
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=300,
        early_stopping=False,  # Disabled due to sklearn bug with string labels
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print(f"  ✓ TRAINING COMPLETE")
    print(f"  Test Accuracy: {accuracy*100:.1f}%")
    print("="*60)
    
    # Show per-class accuracy
    print("\nPer-letter accuracy:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'classes': sorted(list(set(y))),
        'img_size': IMG_SIZE,
        'accuracy': accuracy,
        'n_samples': len(X)
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to: {MODEL_PATH}")
    print("\n🔄 Restart the backend (python app.py) to use the new model!")
    
    return accuracy

def main():
    print("\n" + "="*70)
    print("      SIGN LANGUAGE MODEL - MANUAL TRAINING TOOL")
    print("="*70)
    
    while True:
        print("""
OPTIONS:
  1. Collect YOUR data + Train (RECOMMENDED - best accuracy for YOU)
  2. Train on existing public/ images only  
  3. Train on BOTH your data + public images
  4. Exit
        """)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            print("\n→ Starting data collection...")
            print("   Press letter keys to collect, 'T' to train, 'Q' to quit")
            input("   Press ENTER when ready...")
            if collect_data():
                train_model(use_my_data=True, use_public_data=False)
        
        elif choice == '2':
            train_model(use_my_data=False, use_public_data=True)
        
        elif choice == '3':
            print("\n→ Starting data collection...")
            input("   Press ENTER when ready...")
            if collect_data():
                train_model(use_my_data=True, use_public_data=True)
        
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice, try again.")

if __name__ == '__main__':
    main()
