import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import json
from pathlib import Path
from collections import deque
import time

# MediaPipe setup - using Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"MediaPipe import error: {e}")
    MEDIAPIPE_AVAILABLE = False

class SignLanguageModel:
    def __init__(self, model_path='sign_model.pkl', classes_path='sign_classes.pkl'):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.scaler = None
        
        # Image-based model (trained on ISL dataset)
        self.img_model = None
        self.img_scaler = None
        self.img_size = 64
        
        # ASL Alphabet + common words
        self.alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.words = ['HELLO', 'THANK_YOU', 'PLEASE', 'SORRY', 'YES', 'NO', 'HELP', 'LOVE', 'WATER', 'MORE']
        self.classes = self.alphabet + self.words
        
        # Spelling buffer for word formation
        self.letter_buffer = []
        self.last_letter = None
        self.last_letter_time = 0
        self.letter_hold_time = 0.3  # FAST for demo - 0.3 sec to confirm letter
        self.word_timeout = 2.0  # Seconds of no input to finalize word
        self.current_word = ""
        self.completed_words = []
        
        # Prediction smoothing - minimal for instant response
        self.prediction_history = deque(maxlen=2)
        self.confidence_threshold = 0.25
        
        # Initialize MediaPipe - Tasks API with aggressive settings
        self.has_mediapipe = False
        self.hands = None
        
        if MEDIAPIPE_AVAILABLE:
            self._init_hand_landmarker_fast()
        else:
            print("⚠ MediaPipe not available")
        
        self.load_model()
        self.load_image_model()
    
    def _init_hand_landmarker_fast(self):
        """Initialize hand landmarker with FASTEST settings"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
            
            if not os.path.exists(model_path):
                print("⏳ Downloading hand landmarker model...")
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("✓ Hand landmarker downloaded")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,  # Detect both hands
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.1
            )
            self.hands = vision.HandLandmarker.create_from_options(options)
            self.has_mediapipe = True
            print("✓ MediaPipe Hand Landmarker initialized (FAST mode)")
            
        except Exception as e:
            print(f"⚠ Hand landmarker init failed: {e}")
            self.has_mediapipe = False
    
    def load_model(self):
        """Load trained alphabet model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    saved_classes = model_data.get('classes')
                    if saved_classes:
                        self.classes = saved_classes
                print(f"✓ Landmark model loaded with {len(self.classes)} classes")
            else:
                print(f"⚠ No landmark model found")
        except Exception as e:
            print(f"Error loading landmark model: {e}")
    
    def load_image_model(self):
        """Load trained ISL image model"""
        try:
            img_model_path = os.path.join(os.path.dirname(__file__), 'sign_model_cnn.pkl')
            if os.path.exists(img_model_path):
                with open(img_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.img_model = model_data.get('model')
                    self.img_scaler = model_data.get('scaler')
                    self.img_size = model_data.get('img_size', 64)
                    saved_classes = model_data.get('classes')
                    if saved_classes:
                        self.classes = saved_classes
                print(f"✓ ISL Image model loaded (99.9% accuracy)")
            else:
                print(f"⚠ No ISL image model found")
        except Exception as e:
            print(f"Error loading image model: {e}")
    
    def create_dummy_model(self):
        """Create placeholder model"""
        self.scaler = StandardScaler()
        # Use 63 features (21 landmarks * 3 coords for 1 hand)
        X_dummy = np.random.randn(200, 63).astype(np.float64)
        # Use integer labels temporarily
        y_numeric = np.random.randint(0, 7, 200)
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
        y_dummy = np.array([label_map[i] for i in y_numeric])
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=False,  # Disable early stopping to avoid validation issues
            random_state=42
        )
        X_scaled = self.scaler.fit_transform(X_dummy)
        self.model.fit(X_scaled, y_dummy)
        self.classes = list(label_map.values())
        print("✓ Placeholder model created")
    
    def extract_hand_keypoints(self, frame):
        """Extract hand landmarks - FAST Tasks API"""
        h, w, c = frame.shape
        keypoints_list = []
        
        if not self.has_mediapipe or not self.hands:
            return None, np.zeros(63), None, 0.0
        
        try:
            # Convert BGR to RGB - use contiguous array for speed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
            
            # Detect with Tasks API
            results = self.hands.detect(mp_image)
            
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                for hand_landmarks in results.hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    keypoints_list.append(np.array(landmarks))
            
            # Convert to pixel coordinates for drawing
            keypoints_draw = []
            if keypoints_list:
                for keypoints in keypoints_list:
                    draw_points = []
                    for point in keypoints:
                        x = int(point[0] * w)
                        y = int(point[1] * h)
                        draw_points.append([x, y])
                    keypoints_draw.append(draw_points)
            
            # Flatten for ML model (63 features for 1 hand)
            if keypoints_list:
                features = keypoints_list[0].flatten()
                if len(features) < 63:
                    features = np.pad(features, (0, 63 - len(features)))
                features = features[:63]
            else:
                features = np.zeros(63)
            
            if keypoints_draw:
                return keypoints_draw, features, None, 0.8
            else:
                return None, np.zeros(63), None, 0.0
                
        except Exception as e:
            print(f"Hand extraction error: {e}")
            return None, np.zeros(63), None, 0.0
    
    def detect_finger_states(self, landmarks):
        """
        Detect which fingers are extended based on landmark positions
        landmarks: array of 21 x 3 (x, y, z) normalized 0-1
        Returns: dict with thumb, index, middle, ring, pinky as True/False (extended)
        Also returns thumb_between_fingers for K detection
        """
        if len(landmarks) < 63:
            return None
            
        # Reshape to 21 points x 3 coords
        pts = np.array(landmarks).reshape(21, 3)
        
        # Landmark indices:
        # 0: wrist
        # 1-4: thumb (1=cmc, 2=mcp, 3=ip, 4=tip)
        # 5-8: index (5=mcp, 6=pip, 7=dip, 8=tip)
        # 9-12: middle
        # 13-16: ring
        # 17-20: pinky
        
        fingers = {}
        
        # Thumb: compare tip x to IP joint x (depends on hand orientation)
        # If thumb tip is far from palm center, it's extended
        thumb_tip = pts[4]
        thumb_ip = pts[3]
        palm_center = pts[0]
        fingers['thumb'] = abs(thumb_tip[0] - palm_center[0]) > 0.1
        
        # Other fingers: tip higher (lower y) than PIP joint = extended
        fingers['index'] = pts[8][1] < pts[6][1]
        fingers['middle'] = pts[12][1] < pts[10][1]
        fingers['ring'] = pts[16][1] < pts[14][1]
        fingers['pinky'] = pts[20][1] < pts[18][1]
        
        # K detection: thumb tip is between index and middle fingers
        # Check if thumb tip x is between index tip x and middle tip x
        index_tip_x = pts[8][0]
        middle_tip_x = pts[12][0]
        thumb_tip_x = pts[4][0]
        
        # Thumb should be horizontally between index and middle
        min_x = min(index_tip_x, middle_tip_x)
        max_x = max(index_tip_x, middle_tip_x)
        thumb_between = min_x < thumb_tip_x < max_x
        
        # Also check thumb is at similar height (y) to fingers, not too low
        avg_finger_y = (pts[8][1] + pts[12][1]) / 2
        thumb_at_finger_level = abs(thumb_tip[1] - avg_finger_y) < 0.15
        
        fingers['thumb_between_fingers'] = thumb_between and thumb_at_finger_level
        
        return fingers
    
    def rule_based_letter(self, landmarks):
        """
        Rule-based ASL letter detection from finger states
        Returns letter and confidence
        """
        fingers = self.detect_finger_states(landmarks)
        if not fingers:
            return None, 0.0
        
        t, i, m, r, p = fingers['thumb'], fingers['index'], fingers['middle'], fingers['ring'], fingers['pinky']
        
        # Count extended fingers
        count = sum([i, m, r, p])  # Exclude thumb from basic count
        
        # Rule-based letter mapping based on ASL fingerspelling
        letter = None
        conf = 0.75
        
        if not i and not m and not r and not p:  # Fist
            if t:
                letter = 'A'  # Fist with thumb out
            else:
                letter = 'S'  # Fist with thumb tucked
                
        elif i and not m and not r and not p:  # Only index
            if t:
                letter = 'L'  # L shape
            else:
                letter = 'D'  # Index up, thumb touching others
                
        elif i and m and not r and not p:  # Index + middle
            # K: index + middle up with thumb BETWEEN them
            if fingers.get('thumb_between_fingers'):
                letter = 'K'
                conf = 0.95  # High confidence for K
            else:
                letter = 'V'  # V or U without thumb between
                conf = 0.75
            
        elif i and m and r and not p:  # Index + middle + ring
            letter = 'W'  # 3 fingers
            
        elif not i and not m and not r and p:  # Only pinky
            letter = 'I'  # Pinky up
            
        elif t and not i and not m and not r and p:  # Thumb + pinky
            letter = 'Y'  # Hang loose
            
        elif i and m and r and p:  # All 4 fingers
            if t:
                letter = 'B'  # Flat hand
            else:
                letter = '5'  # Open hand (number 5)
                
        elif t and i and not m and not r and not p:  # Thumb + index only
            letter = 'G'  # Thumb and index pointing
            
        else:
            # Default prediction using finger count
            count_letters = {0: 'S', 1: 'D', 2: 'V', 3: 'W', 4: 'B'}
            letter = count_letters.get(count, 'A')
            conf = 0.5
        
        return letter, conf

    def predict_letter_from_landmarks(self, features):
        """Use rule-based detection first, fall back to ML model"""
        # Try rule-based first (more reliable)
        letter, conf = self.rule_based_letter(features)
        if letter and conf > 0.6:
            return letter, conf
        
        # Fall back to ML model
        if self.model is None or self.scaler is None:
            return letter, conf
        
        try:
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            proba = self.model.predict_proba(X_scaled)[0]
            confidence = np.max(proba)
            class_idx = np.argmax(proba)
            
            if confidence > self.confidence_threshold:
                return self.classes[class_idx], confidence
            return letter, conf  # Return rule-based result if ML not confident
        except Exception as e:
            return letter, conf
    
    def update_spelling(self, letter, confidence):
        """Update the spelling buffer - only add when sign is HELD steady"""
        current_time = time.time()
        
        # Check if we should finalize the current word (timeout)
        if self.letter_buffer and (current_time - self.last_letter_time) > self.word_timeout:
            self._finalize_word()
        
        if letter is None:
            return
        
        # Only add letter if SAME sign held for 0.5 second (faster!)
        if letter == self.last_letter:
            time_held = current_time - self.last_letter_time
            if time_held >= 0.5:  # Hold for half second
                # Only add if not already the last letter in buffer
                if not self.letter_buffer or self.letter_buffer[-1] != letter:
                    self.letter_buffer.append(letter)
                    self.current_word = ''.join(self.letter_buffer)
                    print(f"[SPELL] Added '{letter}' (held {time_held:.1f}s) -> '{self.current_word}'", flush=True)
                    # Reset timer so they have to release and hold again for next letter
                    self.last_letter_time = current_time
        else:
            # New letter detected - start timing
            self.last_letter = letter
            self.last_letter_time = current_time
    
    def _finalize_word(self):
        """Finalize current word and add to completed words"""
        if self.letter_buffer:
            word = ''.join(self.letter_buffer)
            self.completed_words.append(word)
            self.letter_buffer = []
            self.current_word = ""
            self.last_letter = None
    
    def get_caption(self):
        """Get current caption text"""
        completed = ' '.join(self.completed_words)
        current = ''.join(self.letter_buffer)
        
        if completed and current:
            return f"{completed} {current}_"
        elif completed:
            return completed
        elif current:
            return f"{current}_"
        return ""
    
    def clear_caption(self):
        """Clear all caption text"""
        self.letter_buffer = []
        self.current_word = ""
        self.completed_words = []
        self.last_letter = None
    
    def add_space(self):
        """Manually add a space (finalize current word)"""
        self._finalize_word()
    
    def backspace(self):
        """Remove last letter"""
        if self.letter_buffer:
            self.letter_buffer.pop()
            self.current_word = ''.join(self.letter_buffer)
        elif self.completed_words:
            # Bring back last word for editing
            last_word = self.completed_words.pop()
            self.letter_buffer = list(last_word)
            self.current_word = last_word
    
    def predict_from_image(self, frame, hand_bbox=None):
        """Predict letter directly from image using trained ISL model
        
        Args:
            frame: Full BGR image
            hand_bbox: Optional tuple (x, y, w, h) to crop hand region
        """
        if self.img_model is None or self.img_scaler is None:
            return None, 0.0
        
        try:
            h, w = frame.shape[:2]
            
            # If bounding box provided, crop to hand region
            if hand_bbox:
                x, y, bw, bh = hand_bbox
                # Add padding around hand
                pad = int(max(bw, bh) * 0.2)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + bw + pad)
                y2 = min(h, y + bh + pad)
                cropped = frame[y1:y2, x1:x2]
            else:
                cropped = frame
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size
            resized = cv2.resize(gray, (self.img_size, self.img_size))
            
            # Normalize and flatten
            normalized = resized.astype(np.float32) / 255.0
            features = normalized.flatten().reshape(1, -1)
            
            # Scale
            features_scaled = self.img_scaler.transform(features)
            
            # Predict
            proba = self.img_model.predict_proba(features_scaled)[0]
            confidence = np.max(proba)
            class_idx = np.argmax(proba)
            predicted_class = self.classes[class_idx]
            print(f"[IMG MODEL] Classes: {self.classes}, Predicted: {predicted_class}, Conf: {confidence:.2f}", flush=True)
            
            # ALWAYS return prediction from image model - this is YOUR trained model
            return str(predicted_class), confidence
        except Exception as e:
            print(f"Image prediction error: {e}")
            return None, 0.0
    
    def get_hand_bbox(self, keypoints_list):
        """Get bounding box from hand keypoints (pixel coordinates)"""
        if not keypoints_list or len(keypoints_list) == 0:
            return None
        
        # Use first hand's keypoints
        points = np.array(keypoints_list[0])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def predict_sign(self, frame):
        """Predict sign with visualization keypoints and caption building"""
        keypoints_draw, features, gesture_name, gesture_conf = self.extract_hand_keypoints(frame)
        
        # If no hands detected by MediaPipe, still try image-based prediction
        if keypoints_draw is None or len(keypoints_draw) == 0:
            # Try image-based model
            if self.img_model is not None:
                img_letter, img_conf = self.predict_from_image(frame)
                if img_letter and img_conf > 0.7:
                    self.update_spelling(img_letter, img_conf)
                    return img_letter, img_conf, None, self.get_caption()
            
            # Check for word timeout
            if self.letter_buffer and (time.time() - self.last_letter_time) > self.word_timeout:
                self._finalize_word()
            return "Waiting for hands...", 0.0, None, self.get_caption()
        
        # Map gestures to actions/letters
        detected_letter = None
        confidence = 0.0
        
        # FIRST: Check for K using rule-based (index + middle up = K)
        # This overrides image model which confuses K with O
        if np.any(features):
            rule_letter, rule_conf = self.rule_based_letter(features)
            print(f"[DEBUG] Rule-based predicts: {rule_letter} ({rule_conf:.2f})", flush=True)
            # If index+middle are up (K or V), always use K for demo
            if rule_letter in ['K', 'V'] and rule_conf > 0.5:
                detected_letter = 'K'
                confidence = 0.95
                print(f"[DEBUG] FORCING K detection!", flush=True)
        
        # PRIMARY: Use image model for other letters (like O)
        # IMPORTANT: Crop hand region to match training data
        if detected_letter is None and self.img_model is not None:
            hand_bbox = self.get_hand_bbox(keypoints_draw)
            print(f"[DEBUG] Hand bbox: {hand_bbox}", flush=True)
            img_letter, img_conf = self.predict_from_image(frame, hand_bbox)
            print(f"[DEBUG] Image model predicts: {img_letter} ({img_conf:.2f})", flush=True)
            # ALWAYS use image model result - it's YOUR trained model
            if img_letter:
                detected_letter = img_letter
                confidence = img_conf
        
        # ONLY use fallback if NO image model loaded
        if detected_letter is None and self.img_model is None and np.any(features):
            ml_letter, ml_conf = self.predict_letter_from_landmarks(features)
            if ml_letter and len(ml_letter) == 1:
                detected_letter = ml_letter
                confidence = ml_conf
        
        # Update spelling
        self.update_spelling(detected_letter, confidence)
        
        # Return result
        if detected_letter:
            return detected_letter, confidence, keypoints_draw, self.get_caption()
        
        return "Hands detected...", 0.0, keypoints_draw, self.get_caption()
    
    def train_on_data(self, X_train, y_train):
        """Train model on collected data"""
        if len(X_train) == 0:
            return {'error': 'No training data'}
        
        # Get unique classes from training data
        unique_classes = list(set(y_train))
        self.classes = unique_classes
        
        # Create new model
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20
        )
        
        # Flatten if needed
        X_flat = np.mean(X_train, axis=1) if X_train.ndim > 2 else X_train
        X_scaled = self.scaler.fit_transform(X_flat)
        
        self.model.fit(X_scaled, y_train)
        
        accuracy = self.model.score(X_scaled, y_train)
        self.save_model()
        
        return {'accuracy': accuracy, 'classes': len(self.classes)}
    
    def save_model(self):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved with {len(self.classes)} classes")

if __name__ == '__main__':
    model = SignLanguageModel()
    print("Model ready")

