import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
from datetime import datetime
import os

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False


class HandSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Sign Language Recognition")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a1a")
        
        # State
        self.camera_active = False
        self.last_sign = "Waiting for hands..."
        self.confidence = 0
        self.transcript = ""
        self.cap = None
        self.hands = None
        self.last_keypoints = []
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            try:
                # Create hand landmarker for pose detection
                base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self.hands = vision.HandLandmarker.create_from_options(options)
            except:
                # Fallback to older API if available
                try:
                    from mediapipe.framework.formats import landmark_pb2
                    self.hands = None
                    MEDIAPIPE_AVAILABLE = False
                except:
                    MEDIAPIPE_AVAILABLE = False
        
        # Setup GUI
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        """Setup the UI components"""
        # Top bar
        top_frame = tk.Frame(self.root, bg="#0f0f0f", height=60)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(top_frame, text="🤟 Hand Sign Language Recognition", 
                font=("Arial", 20, "bold"), fg="#00ff96", bg="#0f0f0f").pack(side=tk.LEFT, padx=20, pady=10)
        
        # Main content
        content = tk.Frame(self.root, bg="#1a1a1a")
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video
        left_frame = tk.Frame(content, bg="#1a1a1a")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video label
        self.video_label = tk.Label(left_frame, bg="#000", width=640, height=480)
        self.video_label.pack(padx=5, pady=5)
        
        # Detection info
        self.status_label = tk.Label(left_frame, text="Ready", 
                                     font=("Arial", 14, "bold"), fg="#00ff96", bg="#1a1a1a")
        self.status_label.pack(pady=5)
        
        # Confidence
        self.confidence_label = tk.Label(left_frame, text="Confidence: 0%", 
                                         font=("Arial", 12), fg="#aaa", bg="#1a1a1a")
        self.confidence_label.pack()
        
        self.confidence_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(left_frame, variable=self.confidence_var, 
                                        maximum=100, length=300, mode='determinate')
        self.progress.pack(pady=5)
        
        # Right side - Info panel
        right_frame = tk.Frame(content, bg="#0f0f0f", width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        # Controls
        tk.Label(right_frame, text="CONTROLS", font=("Arial", 14, "bold"), 
                fg="#00ff96", bg="#0f0f0f").pack(pady=10)
        
        self.start_btn = tk.Button(right_frame, text="▶ START CAMERA", 
                                   command=self.start_camera, font=("Arial", 12, "bold"),
                                   bg="#00ff96", fg="#000", width=20, height=2)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(right_frame, text="⏹ STOP CAMERA", 
                                  command=self.stop_camera, font=("Arial", 12, "bold"),
                                  bg="#ff4444", fg="#fff", width=20, height=2, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
        tk.Button(right_frame, text="🗑 CLEAR TRANSCRIPT", 
                 command=self.clear_transcript, font=("Arial", 11),
                 bg="#444", fg="#fff", width=20).pack(pady=5)
        
        tk.Button(right_frame, text="💾 SAVE TRANSCRIPT", 
                 command=self.save_transcript, font=("Arial", 11),
                 bg="#444", fg="#fff", width=20).pack(pady=5)
        
        # Detected sign display
        tk.Label(right_frame, text="DETECTED SIGN", font=("Arial", 12, "bold"), 
                fg="#00ff96", bg="#0f0f0f").pack(pady=(20, 5))
        
        self.sign_label = tk.Label(right_frame, text="Waiting for hands...", 
                                   font=("Arial", 16, "bold"), fg="#00ff96", 
                                   bg="#1a1a1a", height=3, relief=tk.SUNKEN)
        self.sign_label.pack(pady=5, padx=5, fill=tk.X)
        
        # Transcript
        tk.Label(right_frame, text="TRANSCRIPT", font=("Arial", 12, "bold"), 
                fg="#00ff96", bg="#0f0f0f").pack(pady=(20, 5))
        
        self.transcript_text = tk.Text(right_frame, height=15, width=40, 
                                       bg="#000", fg="#00ff96", font=("Courier", 10),
                                       relief=tk.SUNKEN)
        self.transcript_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
    
    def start_camera(self):
        """Start the camera feed"""
        self.camera_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="🟢 CAMERA ACTIVE")
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        threading.Thread(target=self.video_loop, daemon=True).start()
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready")
    
    def video_loop(self):
        """Main video processing loop"""
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Preprocess frame for better detection
            frame_bright = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
            frame_rgb = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = self.hands.process(frame_rgb)
            
            # Draw skeleton
            if results.multi_hand_landmarks:
                keypoints = []
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        hand_points.append((x, y))
                    keypoints.append(hand_points)
                
                # Draw hand skeleton
                self.draw_hand_skeleton(frame, keypoints)
                
                # Detect gestures
                gesture = self.detect_gesture(keypoints)
                if gesture:
                    self.last_sign = gesture
                    self.confidence = 95
                    # Add to transcript
                    if not self.transcript or not self.transcript.strip().endswith(gesture):
                        self.transcript += " " + gesture if self.transcript else gesture
                else:
                    self.last_sign = "Hands detected..."
                    self.confidence = 0
            else:
                self.last_sign = "Waiting for hands..."
                self.confidence = 0
            
            # Update UI
            self.update_ui(frame)
    
    def draw_hand_skeleton(self, frame, keypoints):
        """Draw hand skeleton on frame"""
        # MediaPipe hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for hand in keypoints:
            # Draw connections
            for start, end in connections:
                if start < len(hand) and end < len(hand):
                    cv2.line(frame, hand[start], hand[end], (0, 255, 150), 3)
            
            # Draw keypoints
            for point in hand:
                cv2.circle(frame, point, 6, (0, 255, 150), -1)
    
    def detect_gesture(self, keypoints):
        """Detect gestures from hand keypoints"""
        if len(keypoints) < 1:
            return None
        
        # Heart gesture - both hands close together
        if len(keypoints) >= 2:
            hand1 = np.array(keypoints[0])
            hand2 = np.array(keypoints[1])
            palm1 = hand1[9]
            palm2 = hand2[9]
            distance = np.sqrt((palm1[0] - palm2[0])**2 + (palm1[1] - palm2[1])**2)
            if distance < 100:
                return "LOVE"
        
        # Peace sign - index and middle finger up
        hand = np.array(keypoints[0])
        if len(hand) >= 20:
            index_tip = hand[8][1]
            index_pip = hand[6][1]
            middle_tip = hand[12][1]
            middle_pip = hand[10][1]
            
            if (index_tip < index_pip - 20 and middle_tip < middle_pip - 20):
                return "HELLO"
        
        return None
    
    def update_ui(self, frame):
        """Update UI with frame and status"""
        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Update status
        self.sign_label.config(text=self.last_sign)
        self.confidence_label.config(text=f"Confidence: {self.confidence}%")
        self.confidence_var.set(self.confidence)
        
        # Update transcript display
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.delete('1.0', tk.END)
        self.transcript_text.insert('1.0', self.transcript)
        self.transcript_text.see(tk.END)
        self.transcript_text.config(state=tk.DISABLED)
    
    def clear_transcript(self):
        """Clear transcript"""
        self.transcript = ""
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.delete('1.0', tk.END)
        self.transcript_text.config(state=tk.DISABLED)
    
    def save_transcript(self):
        """Save transcript to file"""
        if not self.transcript:
            messagebox.showwarning("Empty", "No transcript to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(self.transcript)
        
        messagebox.showinfo("Saved", f"Transcript saved to {filename}")
    
    def on_closing(self):
        """Handle window close"""
        if self.camera_active:
            self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandSignApp(root)
    root.mainloop()
