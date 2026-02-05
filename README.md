# 🤟 Sign Language Recognition System

A complete machine learning-powered system for real-time sign language recognition that detects **words**, not individual letters. Features both a web interface and desktop application.

## Features

✅ **Word-Level Detection** - Recognizes complete signs (e.g., "APPLE", "HELLO") instead of letter-by-letter
✅ **Real-time Processing** - 30-frame LSTM sequences for accurate word detection
✅ **Multiple Interfaces**:

- Web UI (Vite + React)
- Desktop App (Tkinter + Python)
- REST API (Flask backend)

✅ **MediaPipe Integration** - Robust hand landmark extraction
✅ **TensorFlow/Keras** - LSTM neural network for sequence classification
✅ **Easy Training** - Add new words to vocabulary and retrain

## Architecture

```
Hackathon/
├── backend/
│   ├── sign_model.py      # ML model with LSTM
│   ├── app.py             # Flask REST API server
│   └── requirements.txt
├── src/
│   ├── App.jsx            # React web interface
│   └── App.css
├── tkinter_app.py         # Desktop application
├── ml-model/              # Trained model files directory
└── README.md
```

## Installation

### 1. Backend Setup (Python)

```bash
cd Hackathon/backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install flask flask-cors tensorflow mediapipe opencv-python numpy pillow
```

### 2. Frontend Setup (Node.js)

```bash
cd Hackathon

# Install Node dependencies
npm install

# Dependencies installed:
# - Vite 7.3.1
# - React 18.2
# - ReactDOM 18.2
```

## Quick Start

### Option 1: Web Interface + Flask Backend

**Terminal 1 - Start Flask Backend:**

```bash
cd Hackathon/backend
python app.py
```

Backend starts on `http://localhost:5000`

**Terminal 2 - Start React Frontend:**

```bash
cd Hackathon
npm run dev
```

Frontend available on `http://localhost:5173`

### Option 2: Desktop App

```bash
cd Hackathon
pip install pillow tkinter mediapipe tensorflow opencv-python numpy
python tkinter_app.py
```

## How It Works

### 1. Hand Detection

- MediaPipe extracts 21 hand landmarks per hand
- Each landmark has x, y, z coordinates
- Total: 42 landmarks × 3 coordinates = **126 features per frame**

### 2. Sequence Processing

- System collects **30 consecutive frames**
- Creates 126 × 30 = **3,780-dimensional feature vector**
- LSTM processes this temporal sequence

### 3. Sign Recognition

- LSTM network classifies the entire 30-frame sequence
- Returns predicted word + confidence score
- Only high-confidence predictions (>70%) added to transcript

### 4. Transcript Generation

- Real-time caption display
- Auto-update with recognized words
- Download as .txt file

## API Endpoints

### `GET /health`

Check backend status

```json
{ "status": "ok", "model": "ready" }
```

### `POST /predict`

Predict sign from image

```json
Request:
{"image": "data:image/jpeg;base64,..."}

Response:
{
  "word": "HELLO",
  "confidence": 0.95,
  "words": ["HELLO", "THANK_YOU", "APPLE", ...]
}
```

### `GET /classes`

Get available sign words

```json
{ "classes": ["HELLO", "THANK_YOU", "APPLE", "WATER", "MORE", "HELP"] }
```

### `POST /reset`

Reset prediction sequence buffer

```json
{ "status": "reset" }
```

### `POST /train`

Train model with new data

```json
Request:
{
  "X_train": [...],
  "y_train": [...],
  "epochs": 10
}

Response:
{"status": "trained", "accuracy": 0.95}
```

## Training Custom Signs

### Step 1: Collect Data

```python
import cv2
import numpy as np
import mediapipe as mp
from sign_model import SignLanguageModel

# Initialize
cap = cv2.VideoCapture(0)
model = SignLanguageModel()

# Collect sequences for training
sequences = []
labels = []

# Record 30 frames for each sign
```

### Step 2: Add to Vocabulary

Edit `backend/sign_model.py` and update default classes:

```python
self.classes = ['HELLO', 'THANK_YOU', 'APPLE', 'WATER', 'MORE', 'HELP', 'YOUR_NEW_WORD']
```

### Step 3: Train Model

```python
# After collecting sequences
X_train = np.array(sequences)  # Shape: (samples, 30, 126)
y_train = np.array(labels)     # Shape: (samples,)

model.train_on_data(X_train, y_train, epochs=20)
```

## Default Vocabulary

The system comes pre-configured to recognize:

- **HELLO** - Standard hello gesture
- **THANK_YOU** - Gratitude sign
- **APPLE** - Object sign for apple
- **WATER** - Liquid sign
- **MORE** - "More/again" sign
- **HELP** - Request for assistance

Add more words by extending the `classes` list and retraining.

## Model Architecture

```
Input Layer: (None, 30, 126)
    ↓
LSTM Layer 1: 64 units, return_sequences=True
    ↓
Dropout: 0.5
    ↓
LSTM Layer 2: 128 units
    ↓
Dropout: 0.5
    ↓
Dense: number_of_classes units
    ↓
Softmax Output: Probability distribution
```

## Files & Functions

### sign_model.py

**Class: `SignLanguageModel`**

Methods:

- `__init__()` - Initialize model and load vocabulary
- `load_model()` - Load trained Keras model or create dummy
- `extract_landmarks(frame)` - Extract 126 features from image
- `predict_sign(frame)` - Process frame, buffer 30 frames, predict
- `create_dummy_model()` - Create new LSTM from scratch
- `train_on_data(X_train, y_train, epochs)` - Train on sequences
- `save_classes()` - Persist vocabulary to pickle file

### app.py

**Framework: Flask with CORS**

Routes:

- `@app.route('/health', methods=['GET'])` - Status check
- `@app.route('/predict', methods=['POST'])` - Image to word
- `@app.route('/classes', methods=['GET'])` - List words
- `@app.route('/reset', methods=['POST'])` - Reset buffer
- `@app.route('/train', methods=['POST'])` - Retrain model

### tkinter_app.py

**GUI: Tkinter desktop application**

Features:

- Live video feed with hand landmark visualization
- Real-time sign detection display
- Confidence score percentage
- Transcription output with text editing
- Save transcription to file
- Clear/reset buttons

### App.jsx

**Framework: React + Vite**

Features:

- Video capture with canvas
- Frame-by-frame sending to backend
- Real-time transcription display
- Confidence percentage
- Download transcript

## Troubleshooting

### Flask Backend Won't Start

```bash
# Check port 5000 is available
netstat -ano | findstr :5000  # Windows
lsof -i :5000  # Mac/Linux

# If occupied, change port in app.py
```

### Camera Not Working

```bash
# Install camera driver
# Check browser permissions
# Try different camera index: cv2.VideoCapture(1)
```

### CORS Errors

Ensure Flask-CORS is installed:

```bash
pip install flask-cors
```

### Model Not Loading

```bash
# Check sign_model.h5 exists in ml-model/
# Delete and retrain:
os.remove('ml-model/sign_model.h5')
model.create_dummy_model()
```

## Performance Tips

1. **Good Lighting** - Ensure well-lit environment
2. **Clear Hands** - Keep hands visible and unobstructed
3. **Smooth Motion** - Perform signs slowly and deliberately
4. **Confidence Threshold** - Adjust in `app.py` (default 0.7)
5. **Frame Skip** - Change frame interval in `App.jsx` (default 5)

## Dependencies

- **Python 3.8+**
- TensorFlow 2.x
- MediaPipe 0.8.8+
- Flask 2.x
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

- **Node.js 16+**
- React 18.2
- Vite 7.3.1

## License

MIT License - Feel free to use and modify

## Support

For issues or questions:

1. Check troubleshooting section
2. Verify all dependencies installed
3. Check console for error messages
4. Ensure Flask backend running before web UI

---

**Happy Signing!** 🤟

Made with ❤️ for accessible communication

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
