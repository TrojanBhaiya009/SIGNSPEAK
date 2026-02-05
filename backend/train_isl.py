"""
Train ISL (Indian Sign Language) Model from Image Dataset
Uses direct image features (resized + flattened) for fast training
"""

import os
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
DATA_DIR = Path('../public')  # Contains A/, B/, C/, etc.
MODEL_PATH = 'sign_model_cnn.pkl'
IMG_SIZE = 64  # Resize images to 64x64
SAMPLES_PER_LETTER = 200  # Use more samples for better accuracy

def load_and_preprocess_image(image_path, size=64):
    """Load image, convert to grayscale, resize, and flatten"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize
        resized = cv2.resize(gray, (size, size))
        
        # Normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        # Flatten
        return normalized.flatten()
    except:
        return None

def load_dataset(data_dir, img_size=64, samples_per_letter=200):
    """Load images for each letter"""
    X = []
    y = []
    
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    print("\n📊 Loading ISL images...")
    print("=" * 50)
    
    for letter in alphabet:
        letter_dir = data_dir / letter
        if not letter_dir.exists():
            print(f"  ⚠ Folder {letter} not found, skipping")
            continue
        
        # Get image files
        images = list(letter_dir.glob('*.jpg')) + list(letter_dir.glob('*.png'))
        
        # Limit samples
        if len(images) > samples_per_letter:
            indices = np.linspace(0, len(images)-1, samples_per_letter, dtype=int)
            images = [images[i] for i in indices]
        
        loaded = 0
        for img_path in images:
            features = load_and_preprocess_image(img_path, img_size)
            if features is not None:
                X.append(features)
                y.append(letter)
                loaded += 1
        
        print(f"  {letter}: {loaded} images loaded")
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train neural network classifier"""
    print("\n⏳ Training neural network...")
    print("=" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP (no early stopping to avoid sklearn bug with string labels)
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=50,
        early_stopping=False,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy*100:.1f}%")
    
    return model, scaler

def save_model(model, scaler, classes, path):
    """Save trained model"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'classes': classes,
        'type': 'image',
        'img_size': IMG_SIZE
    }
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✓ Model saved to {path}")

def main():
    print("=" * 50)
    print("🤟 ISL Model Training (Image-based)")
    print("=" * 50)
    
    # Load dataset
    X, y = load_dataset(DATA_DIR, IMG_SIZE, SAMPLES_PER_LETTER)
    
    if len(X) == 0:
        print("❌ No images loaded!")
        return
    
    print(f"\n✓ Total samples: {len(X)}")
    print(f"✓ Feature size: {X.shape[1]}")
    print(f"✓ Classes: {len(np.unique(y))}")
    
    # Train
    model, scaler = train_model(X, y)
    
    # Save
    classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    save_model(model, scaler, classes, MODEL_PATH)
    
    print("\n🎉 Training complete!")

if __name__ == '__main__':
    main()
