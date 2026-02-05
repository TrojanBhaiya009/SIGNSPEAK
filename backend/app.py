from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import sys
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(__file__))
from sign_model import SignLanguageModel

app = Flask(__name__)
CORS(app)

# Initialize ML model
model = SignLanguageModel()

# In-memory meeting storage (in production use database)
meetings = {}
CLERK_SECRET_KEY = 'sk_test_Ys82HYYs19YcjGZxPspK6ZcwRTUi0aV7XQcjRXw1Q5'

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'ready'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with base64 encoded image
    Returns sign, confidence, keypoints for visualization, and caption
    """
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"[PREDICT] Received frame shape: {frame.shape}", flush=True)
        
        # DON'T flip here - frontend handles the mirror display
        # Process original frame orientation
        
        # Predict with keypoints - now returns 4 values including caption
        result = model.predict_sign(frame)
        sign, confidence, keypoints, caption = result if len(result) == 4 else (*result, "")
        
        # Convert keypoints to serializable format
        # Return original coordinates - frontend will handle mirroring
        keypoints_json = []
        if keypoints:
            for hand_points in keypoints:
                keypoints_json.append([[int(p[0]), int(p[1])] for p in hand_points])
        
        return jsonify({
            'sign': sign,
            'confidence': float(confidence),
            'keypoints': keypoints_json,
            'caption': caption
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_landmarks', methods=['POST'])
def predict_landmarks():
    """
    FAST endpoint - receives landmarks directly from client-side MediaPipe
    No image processing needed - just ML prediction
    """
    try:
        data = request.json
        landmarks = data.get('landmarks', [])
        
        if not landmarks or len(landmarks) < 63:
            return jsonify({'sign': 'Unknown', 'confidence': 0, 'caption': model.get_caption()})
        
        # Convert to numpy array
        features = np.array(landmarks[:63], dtype=np.float64)
        
        # Predict letter
        letter, confidence = model.predict_letter_from_landmarks(features)
        
        # Update spelling
        model.update_spelling(letter, confidence)
        
        return jsonify({
            'sign': letter or 'Unknown',
            'confidence': float(confidence),
            'caption': model.get_caption()
        })
    except Exception as e:
        print(f"Landmark prediction error: {e}")
        return jsonify({'sign': 'Unknown', 'confidence': 0, 'caption': ''})

@app.route('/caption/clear', methods=['POST'])
def clear_caption():
    """Clear the current caption"""
    model.clear_caption()
    return jsonify({'success': True})

@app.route('/caption/space', methods=['POST'])
def add_space():
    """Add a space (finalize current word)"""
    model.add_space()
    return jsonify({'success': True, 'caption': model.get_caption()})

@app.route('/caption/backspace', methods=['POST'])
def backspace():
    """Remove last letter"""
    model.backspace()
    return jsonify({'success': True, 'caption': model.get_caption()})

# Meeting management endpoints
@app.route('/meeting/create', methods=['POST'])
def create_meeting():
    """Create a new meeting"""
    try:
        data = request.json
        meeting_code = data.get('code')
        host_id = data.get('host_id')
        host_name = data.get('host_name')
        
        if not meeting_code:
            return jsonify({'error': 'No meeting code'}), 400
        
        meetings[meeting_code] = {
            'code': meeting_code,
            'host_id': host_id,
            'host_name': host_name,
            'created_at': datetime.now().isoformat(),
            'participants': [
                {'id': host_id, 'name': host_name, 'is_host': True, 'joined_at': datetime.now().isoformat()}
            ],
            'pending': []
        }
        
        return jsonify({'success': True, 'meeting': meetings[meeting_code]}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/meeting/<code>/join', methods=['POST'])
def join_meeting(code):
    """Request to join a meeting"""
    try:
        data = request.json
        participant_id = data.get('participant_id')
        participant_name = data.get('participant_name')
        
        if code not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
        
        # Add to pending
        meetings[code]['pending'].append({
            'id': participant_id,
            'name': participant_name,
            'requested_at': datetime.now().isoformat()
        })
        
        return jsonify({'success': True, 'status': 'pending'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/meeting/<code>/admit', methods=['POST'])
def admit_participant(code):
    """Host admits a pending participant"""
    try:
        data = request.json
        participant_id = data.get('participant_id')
        
        if code not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
        
        # Find and remove from pending
        pending = meetings[code]['pending']
        participant = next((p for p in pending if p['id'] == participant_id), None)
        
        if not participant:
            return jsonify({'error': 'Participant not found'}), 404
        
        # Add to participants
        meetings[code]['participants'].append({
            'id': participant_id,
            'name': participant['name'],
            'is_host': False,
            'joined_at': datetime.now().isoformat()
        })
        
        meetings[code]['pending'].remove(participant)
        
        return jsonify({'success': True}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/meeting/<code>/reject', methods=['POST'])
def reject_participant(code):
    """Host rejects a pending participant"""
    try:
        data = request.json
        participant_id = data.get('participant_id')
        
        if code not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meetings[code]['pending'] = [
            p for p in meetings[code]['pending'] 
            if p['id'] != participant_id
        ]
        
        return jsonify({'success': True}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/meeting/<code>', methods=['GET'])
def get_meeting(code):
    """Get meeting details"""
    if code not in meetings:
        return jsonify({'error': 'Meeting not found'}), 404
    
    return jsonify(meetings[code]), 200

@app.route('/meeting/<code>/leave', methods=['POST'])
def leave_meeting(code):
    """Participant leaves meeting"""
    try:
        data = request.json
        participant_id = data.get('participant_id')
        
        if code not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meetings[code]['participants'] = [
            p for p in meetings[code]['participants']
            if p['id'] != participant_id
        ]
        
        return jsonify({'success': True}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available sign language classes/words"""
    return jsonify({'classes': model.classes})

@app.route('/reset', methods=['POST'])
def reset_sequence():
    """Reset prediction sequence"""
    model.sequence = []
    return jsonify({'status': 'sequence reset'})

@app.route('/train', methods=['POST'])
def train_model():
    """Train model on provided data"""
    try:
        data = request.json
        X_train = np.array(data.get('X_train'))
        y_train = np.array(data.get('y_train'))
        epochs = data.get('epochs', 10)
        
        model.train_on_data(X_train, y_train, epochs=epochs)
        
        return jsonify({'status': 'training complete'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Sign Language Backend Server...")
    print("http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
