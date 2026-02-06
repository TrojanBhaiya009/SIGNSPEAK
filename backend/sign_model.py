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
        # Use absolute path relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, model_path)
        self.classes_path = os.path.join(script_dir, classes_path)
        self.model = None
        self.scaler = None
        
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
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4
            )
            self.hands = vision.HandLandmarker.create_from_options(options)
            self.has_mediapipe = True
            print("✓ MediaPipe Hand Landmarker initialized (FAST mode)")
            
        except Exception as e:
            print(f"⚠ Hand landmarker init failed: {e}")
            self.has_mediapipe = False
    
    def load_model(self):
        """Load trained ASL alphabet model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    saved_classes = model_data.get('classes')
                    if saved_classes:
                        self.classes = saved_classes
                print(f"✓ ASL model loaded with {len(self.classes)} classes")
            else:
                print(f"⚠ No ASL model found at {self.model_path}")
        except Exception as e:
            print(f"Error loading ASL model: {e}")
    
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
        Returns: dict with various finger states
        """
        if len(landmarks) < 63:
            return None
            
        # Reshape to 21 points x 3 coords
        pts = np.array(landmarks).reshape(21, 3)
        
        # Landmark indices:
        # 0: wrist
        # 1-4: thumb (1=cmc, 2=mcp, 3=ip, 4=tip)
        # 5-8: index (5=mcp, 6=pip, 7=dip, 8=tip)
        # 9-12: middle (9=mcp, 10=pip, 11=dip, 12=tip)
        # 13-16: ring
        # 17-20: pinky
        
        fingers = {}
        
        # Thumb extended: tip is far from palm center (wrist)
        thumb_tip = pts[4]
        palm_center = pts[0]
        fingers['thumb'] = abs(thumb_tip[0] - palm_center[0]) > 0.12
        
        # Finger extended: tip is significantly higher than MCP joint
        # Use MCP as reference (more reliable than PIP for curved fingers)
        fingers['index'] = pts[8][1] < pts[5][1] - 0.05
        fingers['middle'] = pts[12][1] < pts[9][1] - 0.05
        fingers['ring'] = pts[16][1] < pts[13][1] - 0.05
        fingers['pinky'] = pts[20][1] < pts[17][1] - 0.05
        
        # Detect curved/closed hand (for O, C, E shapes)
        # All fingertips close together and near thumb
        fingertips = [pts[4], pts[8], pts[12], pts[16], pts[20]]
        fingertip_center = np.mean(fingertips, axis=0)
        fingertip_spread = np.std([p[0] for p in fingertips])  # X spread
        
        # O shape: fingertips clustered together (low spread)
        fingers['fingertips_clustered'] = fingertip_spread < 0.08
        
        # Check if fingertips are touching thumb (for O, F)
        thumb_to_index = np.linalg.norm(pts[4] - pts[8])
        thumb_to_middle = np.linalg.norm(pts[4] - pts[12])
        fingers['thumb_touches_index'] = thumb_to_index < 0.08
        fingers['thumb_touches_middle'] = thumb_to_middle < 0.08
        
        # K detection: thumb tip is between index and middle fingers
        index_tip_x = pts[8][0]
        middle_tip_x = pts[12][0]
        thumb_tip_x = pts[4][0]
        
        min_x = min(index_tip_x, middle_tip_x)
        max_x = max(index_tip_x, middle_tip_x)
        thumb_between = min_x < thumb_tip_x < max_x
        
        avg_finger_y = (pts[8][1] + pts[12][1]) / 2
        thumb_at_finger_level = abs(thumb_tip[1] - avg_finger_y) < 0.15
        
        fingers['thumb_between_fingers'] = thumb_between and thumb_at_finger_level
        
        return fingers
    
    def rule_based_letter(self, landmarks):
        """
        Rule-based ASL letter detection from finger states
        Returns letter and confidence - only for high-confidence patterns
        """
        fingers = self.detect_finger_states(landmarks)
        if not fingers:
            return None, 0.0
        
        t, i, m, r, p = fingers['thumb'], fingers['index'], fingers['middle'], fingers['ring'], fingers['pinky']
        
        # Count extended fingers
        count = sum([i, m, r, p])  # Exclude thumb from basic count
        
        # Only return high-confidence rule-based matches
        # Let ML model handle ambiguous cases like O, C, E, F, J, M, N, etc.
        letter = None
        conf = 0.0
        
        # FIRST: Check for O/C shapes (curved hand with fingertips clustered)
        # If fingertips are clustered, let ML model handle it (O, C, E shapes)
        if fingers.get('fingertips_clustered'):
            return None, 0.0  # Let ML model decide
        
        # Check for thumb touching index (F or O shape) - let ML handle
        if fingers.get('thumb_touches_index'):
            return None, 0.0  # Let ML model decide
        
        # Now check clear finger patterns
        if not i and not m and not r and not p:  # All fingers down (fist)
            if t:
                letter = 'A'  # Fist with thumb out
                conf = 0.8
            else:
                letter = 'S'  # Fist with thumb tucked
                conf = 0.8
                
        elif i and not m and not r and not p:  # Only index extended
            if t:
                letter = 'L'  # L shape - index up, thumb out
                conf = 0.85
            else:
                letter = 'D'  # Index up, thumb not out
                conf = 0.65
                
        elif i and m and not r and not p:  # Index + middle extended
            if fingers.get('thumb_between_fingers'):
                letter = 'K'
                conf = 0.95
            else:
                letter = 'V'  # V or U
                conf = 0.7
            
        elif i and m and r and not p:  # Index + middle + ring extended
            letter = 'W'
            conf = 0.85
            
        elif not i and not m and not r and p:  # ONLY pinky truly extended
            # Make sure it's really just pinky (not curved O/C shape)
            letter = 'I'
            conf = 0.85
            
        elif t and not i and not m and not r and p:  # Thumb + pinky only
            letter = 'Y'
            conf = 0.9
            
        elif i and m and r and p:  # All 4 fingers extended
            if t:
                letter = 'B'  # Flat hand with thumb
                conf = 0.8
        
        # Return None for patterns that should use ML model (O, C, E, F, J, M, N, P, Q, R, T, U, X, Z)
        return letter, conf

    def predict_letter_from_landmarks(self, features):
        """Use ML model as the PRIMARY source - user trained it with their webcam"""
        # ALWAYS try ML model first - it was trained on user's actual hand signs
        ml_letter = None
        ml_conf = 0.0
        
        if self.model is not None and self.scaler is not None:
            try:
                X = features.reshape(1, -1)
                X_scaled = self.scaler.transform(X)
                proba = self.model.predict_proba(X_scaled)[0]
                ml_conf = np.max(proba)
                class_idx = np.argmax(proba)
                ml_letter = self.classes[class_idx]
            except Exception as e:
                pass
        
        # Trust the ML model - it was trained on user's own hand signs
        if ml_letter and ml_conf > 0.3:
            return ml_letter, ml_conf
        
        # Only fall back to rule-based for very low confidence
        rule_letter, rule_conf = self.rule_based_letter(features)
        if rule_letter and rule_conf > 0.8:
            return rule_letter, rule_conf
        
        # Return ML result even with lower confidence
        return ml_letter if ml_letter else rule_letter, max(ml_conf, rule_conf)
    
    def update_spelling(self, letter, confidence):
        """Update the spelling buffer - only add when sign is HELD steady"""
        current_time = time.time()
        
        # Check if we should finalize the current word (timeout)
        if self.letter_buffer and (current_time - self.last_letter_time) > self.word_timeout:
            self._finalize_word()
        
        if letter is None:
            return
        
        # Only add letter if SAME sign held for 0.2 second (FAST!)
        if letter == self.last_letter:
            time_held = current_time - self.last_letter_time
            if time_held >= 0.2:  # Hold for 200ms only
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

    def predict_sign(self, frame):
        """Predict ASL sign with visualization keypoints and caption building"""
        keypoints_draw, features, gesture_name, gesture_conf = self.extract_hand_keypoints(frame)
        
        # If no hands detected
        if keypoints_draw is None or len(keypoints_draw) == 0:
            # Check for word timeout
            if self.letter_buffer and (time.time() - self.last_letter_time) > self.word_timeout:
                self._finalize_word()
            return "Waiting for hands...", 0.0, None, self.get_caption()
        
        # Map gestures to ASL letters
        detected_letter = None
        confidence = 0.0
        
        # Use ASL landmark-based prediction
        if np.any(features):
            # Try rule-based first for common letters
            rule_letter, rule_conf = self.rule_based_letter(features)
            
            # Then use ML model for better accuracy
            ml_letter, ml_conf = self.predict_letter_from_landmarks(features)
            
            # Use ML prediction if confident, otherwise use rule-based
            if ml_letter and ml_conf > 0.5:
                detected_letter = ml_letter
                confidence = ml_conf
            elif rule_letter and rule_conf > 0.5:
                detected_letter = rule_letter
                confidence = rule_conf
        
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

