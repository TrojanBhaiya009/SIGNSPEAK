"""
SignSpeak Backend — Meeting Platform with WebRTC Signaling
==========================================================
Flask + Flask-SocketIO backend providing:
  - WebRTC signaling relay (offer / answer / ICE candidates)
  - Meeting room management (create, join, admit, reject, leave)
  - ASL landmark prediction (REST + Socket.IO)
  - In-meeting chat relay
  - Media state broadcasting (mute / camera toggle)
  - Screen-share coordination
  - Model training endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from sign_model import SignLanguageModel

# ======================================================================
# App setup
# ======================================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

try:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='gevent',
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=10_000_000,
    )
except Exception:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=10_000_000,
    )

# ML model (starts untrained)
model = SignLanguageModel()

# In-memory stores
meetings = {}                    # code -> meeting dict
user_sockets = {}                # sid  -> { user_id, meeting_code, username }

# ======================================================================
# REST endpoints
# ======================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': model.get_status(),
        'active_meetings': len(meetings),
    })


@app.route('/model/status', methods=['GET'])
def model_status():
    return jsonify(model.get_status())


@app.route('/predict_landmarks', methods=['POST'])
def predict_landmarks():
    """Predict ASL letter from 63-dim landmark array."""
    try:
        data = request.json
        landmarks = data.get('landmarks', [])
        
        # BUG FIX: Stricter validation of landmark array
        if not landmarks or not isinstance(landmarks, (list, tuple)):
            return jsonify({'sign': None, 'confidence': 0, 'caption': model.get_caption(), 'sign_type': 'letter'})
        
        # BUG FIX: Ensure exactly 63 dimensions
        if len(landmarks) != 63:
            return jsonify({'sign': None, 'confidence': 0, 'caption': model.get_caption(), 'sign_type': 'letter'})

        features = np.array(landmarks[:63], dtype=np.float64)
        
        # BUG FIX: Validate all values are finite
        if not np.all(np.isfinite(features)):
            return jsonify({'sign': None, 'confidence': 0, 'caption': model.get_caption(), 'sign_type': 'letter'})
        
        letter, confidence = model.predict_from_landmarks(features)
        model.update_spelling(letter, confidence)

        # Determine sign type: phrase (multi-char words), digit, or letter
        if letter and letter in {'HELLO', 'STOP', 'I LOVE YOU', 'NO', 'YES', 'BYE'}:
            sign_type = 'phrase'
        elif letter and letter in {'0','1','2','3','4','5','6','7','8','9'}:
            sign_type = 'digit'
        else:
            sign_type = 'letter'

        return jsonify({
            'sign': letter,
            'confidence': float(confidence),
            'caption': model.get_caption(),
            'sign_type': sign_type,
        })
    except Exception as e:
        return jsonify({'sign': None, 'confidence': 0, 'caption': '', 'error': str(e), 'sign_type': 'letter'}), 400


@app.route('/train', methods=['POST'])
def train():
    """Train model on collected landmark data."""
    try:
        data = request.json
        X_train = np.array(data['X_train'], dtype=np.float64)
        y_train = np.array(data['y_train'])
        result = model.train(X_train, y_train)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/caption/clear', methods=['POST'])
def clear_caption():
    model.clear_caption()
    return jsonify({'success': True, 'caption': ''})


@app.route('/caption/space', methods=['POST'])
def add_space():
    model.add_space()
    return jsonify({'success': True, 'caption': model.get_caption()})


@app.route('/caption/backspace', methods=['POST'])
def backspace():
    model.backspace()
    return jsonify({'success': True, 'caption': model.get_caption()})


# ======================================================================
# Socket.IO — connection lifecycle
# ======================================================================

@socketio.on('connect')
def on_connect():
    print(f"[WS] Connected: {request.sid}")


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    info = user_sockets.pop(sid, None)
    if info:
        code = info.get('meeting_code')
        if code and code in meetings:
            meeting = meetings[code]
            meeting['participants'] = [
                p for p in meeting['participants'] if p['sid'] != sid
            ]
            meeting['pending'] = [
                p for p in meeting['pending'] if p['sid'] != sid
            ]

            # Notify room
            socketio.emit('peer-disconnected', {
                'sid': sid,
                'user_id': info.get('user_id'),
                'username': info.get('username'),
            }, room=code)
            socketio.emit('participants-updated', {
                'participants': meeting['participants'],
                'pending': meeting['pending'],
            }, room=code)

            # Garbage-collect empty meetings
            if not meeting['participants']:
                del meetings[code]
                print(f"[MEETING] Deleted empty meeting: {code}")

    print(f"[WS] Disconnected: {sid}")


# ======================================================================
# Socket.IO — Meeting management
# ======================================================================

@socketio.on('create-meeting')
def on_create_meeting(data):
    code = data.get('code', 'SL-' + uuid.uuid4().hex[:8].upper())
    user_id = data.get('user_id')
    username = data.get('username', 'Host')

    host = {
        'sid': request.sid,
        'user_id': user_id,
        'username': username,
        'is_host': True,
        'joined_at': datetime.now().isoformat(),
        'audio': True,
        'video': True,
    }

    meetings[code] = {
        'code': code,
        'host_sid': request.sid,
        'host_id': user_id,
        'created_at': datetime.now().isoformat(),
        'participants': [host],
        'pending': [],
        'chat': [],
    }

    user_sockets[request.sid] = {
        'user_id': user_id,
        'meeting_code': code,
        'username': username,
    }
    join_room(code)

    emit('meeting-created', {
        'success': True,
        'code': code,
        'meeting': meetings[code],
    })
    print(f"[MEETING] Created: {code} by {username}")


@socketio.on('join-meeting')
def on_join_meeting(data):
    code = data.get('code', '').strip()
    user_id = data.get('user_id')
    username = data.get('username', 'Guest')

    if code not in meetings:
        emit('join-error', {'error': 'Meeting not found'})
        return

    pending_entry = {
        'sid': request.sid,
        'user_id': user_id,
        'username': username,
        'requested_at': datetime.now().isoformat(),
    }

    meetings[code]['pending'].append(pending_entry)
    user_sockets[request.sid] = {
        'user_id': user_id,
        'meeting_code': code,
        'username': username,
    }
    join_room(code)

    # Tell the joiner to wait
    emit('waiting-approval', {'code': code})

    # Notify everyone (host will show in pending list)
    socketio.emit('pending-participant', {
        'participant': pending_entry,
        'pending': meetings[code]['pending'],
    }, room=code)

    print(f"[MEETING] {username} requesting to join: {code}")


@socketio.on('admit-participant')
def on_admit(data):
    code = data.get('code')
    participant_sid = data.get('sid')

    if code not in meetings:
        return
    meeting = meetings[code]

    # Only host can admit
    if request.sid != meeting['host_sid']:
        return

    pending = next(
        (p for p in meeting['pending'] if p['sid'] == participant_sid), None
    )
    if not pending:
        return

    meeting['pending'].remove(pending)

    participant = {
        'sid': pending['sid'],
        'user_id': pending['user_id'],
        'username': pending['username'],
        'is_host': False,
        'joined_at': datetime.now().isoformat(),
        'audio': True,
        'video': True,
    }
    meeting['participants'].append(participant)

    # Tell the admitted user they're in
    socketio.emit('admitted', {
        'code': code,
        'participants': meeting['participants'],
    }, to=participant_sid)

    # Update everyone's participant list
    socketio.emit('participants-updated', {
        'participants': meeting['participants'],
        'pending': meeting['pending'],
    }, room=code)

    # Tell every *existing* participant to create an offer to the new peer
    for p in meeting['participants']:
        if p['sid'] != participant_sid:
            socketio.emit('new-peer', {
                'sid': participant_sid,
                'user_id': pending['user_id'],
                'username': pending['username'],
                'should_create_offer': True,
            }, to=p['sid'])

    print(f"[MEETING] Admitted {pending['username']} to {code}")


@socketio.on('reject-participant')
def on_reject(data):
    code = data.get('code')
    participant_sid = data.get('sid')

    if code not in meetings:
        return
    meeting = meetings[code]
    if request.sid != meeting['host_sid']:
        return

    pending = next(
        (p for p in meeting['pending'] if p['sid'] == participant_sid), None
    )
    if pending:
        meeting['pending'].remove(pending)
        socketio.emit('rejected', {'code': code}, to=participant_sid)
        socketio.emit('participants-updated', {
            'participants': meeting['participants'],
            'pending': meeting['pending'],
        }, room=code)


@socketio.on('leave-meeting')
def on_leave(data):
    code = data.get('code')
    sid = request.sid

    if code not in meetings:
        return

    meeting = meetings[code]
    meeting['participants'] = [
        p for p in meeting['participants'] if p['sid'] != sid
    ]
    meeting['pending'] = [
        p for p in meeting['pending'] if p['sid'] != sid
    ]

    user_sockets.pop(sid, None)
    leave_room(code)

    socketio.emit('peer-disconnected', {'sid': sid}, room=code)
    socketio.emit('participants-updated', {
        'participants': meeting['participants'],
        'pending': meeting['pending'],
    }, room=code)

    if not meeting['participants']:
        del meetings[code]


# ======================================================================
# Socket.IO — WebRTC signaling
# ======================================================================

@socketio.on('webrtc-offer')
def on_offer(data):
    """Relay SDP offer to a specific peer."""
    target_sid = data.get('target')
    socketio.emit('webrtc-offer', {
        'offer': data.get('offer'),
        'from_sid': request.sid,
        'from_username': user_sockets.get(request.sid, {}).get('username', ''),
    }, to=target_sid)


@socketio.on('webrtc-answer')
def on_answer(data):
    """Relay SDP answer to a specific peer."""
    target_sid = data.get('target')
    socketio.emit('webrtc-answer', {
        'answer': data.get('answer'),
        'from_sid': request.sid,
    }, to=target_sid)


@socketio.on('ice-candidate')
def on_ice(data):
    """Relay ICE candidate to a specific peer."""
    target_sid = data.get('target')
    socketio.emit('ice-candidate', {
        'candidate': data.get('candidate'),
        'from_sid': request.sid,
    }, to=target_sid)


# ======================================================================
# Socket.IO — Media state / Chat / Screen share / ASL
# ======================================================================

@socketio.on('media-state')
def on_media_state(data):
    """Broadcast mute or camera toggle to the room."""
    code = data.get('code')
    if code not in meetings:
        return

    # Update stored participant state
    for p in meetings[code]['participants']:
        if p['sid'] == request.sid:
            p['audio'] = data.get('audio', p.get('audio', True))
            p['video'] = data.get('video', p.get('video', True))
            break

    socketio.emit('peer-media-state', {
        'sid': request.sid,
        'audio': data.get('audio'),
        'video': data.get('video'),
    }, room=code)


@socketio.on('chat-message')
def on_chat(data):
    """Relay chat message to the room."""
    code = data.get('code')
    if code not in meetings:
        return

    msg = {
        'id': uuid.uuid4().hex[:8],
        'sid': request.sid,
        'username': user_sockets.get(request.sid, {}).get('username', 'Unknown'),
        'text': data.get('text', ''),
        'timestamp': datetime.now().isoformat(),
        'type': data.get('type', 'text'),   # 'text' | 'caption'
    }
    meetings[code]['chat'].append(msg)
    socketio.emit('chat-message', msg, room=code)


@socketio.on('predict-landmarks')
def on_predict_landmarks(data):
    """ASL prediction via Socket (lower latency than REST)."""
    landmarks = data.get('landmarks', [])
    
    # BUG FIX: Stricter validation of landmark array
    if not landmarks or not isinstance(landmarks, (list, tuple)):
        emit('prediction', {'sign': None, 'confidence': 0, 'caption': model.get_caption(), 'sign_type': 'letter'})
        return
    
    # BUG FIX: Ensure exactly 63 dimensions (21 landmarks × 3 coordinates)
    if len(landmarks) != 63:
        emit('prediction', {'sign': None, 'confidence': 0, 'caption': model.get_caption(), 'sign_type': 'letter'})
        return
    
    try:
        # BUG FIX: Validate all values are numeric and finite
        features = np.array(landmarks[:63], dtype=np.float64)
        if not np.all(np.isfinite(features)):
            emit('prediction', {'sign': None, 'confidence': 0, 'caption': model.get_caption(), 'sign_type': 'letter'})
            return
            
        letter, confidence = model.predict_from_landmarks(features)
        model.update_spelling(letter, confidence)

        # Determine sign type: phrase, digit, or letter
        if letter and letter in {'HELLO', 'STOP', 'I LOVE YOU', 'NO', 'YES', 'BYE'}:
            sign_type = 'phrase'
        elif letter and letter in {'0','1','2','3','4','5','6','7','8','9'}:
            sign_type = 'digit'
        else:
            sign_type = 'letter'

        result = {
            'sign': letter,
            'confidence': float(confidence),
            'caption': model.get_caption(),
            'sign_type': sign_type,
        }
        emit('prediction', result)

        # Broadcast caption to the meeting room
        code = user_sockets.get(request.sid, {}).get('meeting_code')
        if code and code in meetings:
            socketio.emit('peer-caption', {
                'sid': request.sid,
                'username': user_sockets.get(request.sid, {}).get('username'),
                'caption': model.get_caption(),
                'sign': letter,
                'confidence': float(confidence),
                'sign_type': sign_type,
            }, room=code)
    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        emit('prediction', {'sign': None, 'confidence': 0, 'caption': model.get_caption(), 'error': str(e), 'sign_type': 'letter'})


@socketio.on('screen-share-started')
def on_screen_share_start(data):
    code = data.get('code')
    if code:
        socketio.emit('peer-screen-share', {
            'sid': request.sid,
            'username': user_sockets.get(request.sid, {}).get('username'),
            'sharing': True,
        }, room=code)


@socketio.on('screen-share-stopped')
def on_screen_share_stop(data):
    code = data.get('code')
    if code:
        socketio.emit('peer-screen-share', {
            'sid': request.sid,
            'sharing': False,
        }, room=code)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  SignSpeak Backend -- Meeting Platform")
    print("  http://localhost:5000")
    status = 'Trained' if model.is_trained else 'Untrained (rule-based only)'
    print(f"  Model: {status}")
    print("=" * 55)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)
