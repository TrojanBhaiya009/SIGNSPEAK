"""
Pre-train ASL Alphabet Model with Accurate Synthetic Landmark Data
Based on the Gerard Aflague ASL Alphabet Chart

MediaPipe Hand Landmarks (21 points):
0: WRIST
1-4: THUMB (CMC, MCP, IP, TIP)
5-8: INDEX (MCP, PIP, DIP, TIP)
9-12: MIDDLE (MCP, PIP, DIP, TIP)
13-16: RING (MCP, PIP, DIP, TIP)
17-20: PINKY (MCP, PIP, DIP, TIP)

Flattened array indices (x, y, z for each):
Wrist: 0-2, Thumb: 3-14, Index: 15-26, Middle: 27-38, Ring: 39-50, Pinky: 51-62
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Landmark index helpers
WRIST = slice(0, 3)
THUMB_CMC = slice(3, 6)
THUMB_MCP = slice(6, 9)
THUMB_IP = slice(9, 12)
THUMB_TIP = slice(12, 15)
INDEX_MCP = slice(15, 18)
INDEX_PIP = slice(18, 21)
INDEX_DIP = slice(21, 24)
INDEX_TIP = slice(24, 27)
MIDDLE_MCP = slice(27, 30)
MIDDLE_PIP = slice(30, 33)
MIDDLE_DIP = slice(33, 36)
MIDDLE_TIP = slice(36, 39)
RING_MCP = slice(39, 42)
RING_PIP = slice(42, 45)
RING_DIP = slice(45, 48)
RING_TIP = slice(48, 51)
PINKY_MCP = slice(51, 54)
PINKY_PIP = slice(54, 57)
PINKY_DIP = slice(57, 60)
PINKY_TIP = slice(60, 63)


def base_hand():
    """Create base hand position with realistic proportions"""
    landmarks = np.zeros(63)
    
    # Wrist at bottom center
    landmarks[WRIST] = [0.5, 0.85, 0]
    
    # Thumb chain (slightly to the side)
    landmarks[THUMB_CMC] = [0.35, 0.75, 0]
    landmarks[THUMB_MCP] = [0.30, 0.68, 0]
    landmarks[THUMB_IP] = [0.28, 0.60, 0]
    landmarks[THUMB_TIP] = [0.27, 0.52, 0]
    
    # Index finger chain
    landmarks[INDEX_MCP] = [0.40, 0.58, 0]
    landmarks[INDEX_PIP] = [0.40, 0.45, 0]
    landmarks[INDEX_DIP] = [0.40, 0.35, 0]
    landmarks[INDEX_TIP] = [0.40, 0.25, 0]
    
    # Middle finger chain
    landmarks[MIDDLE_MCP] = [0.50, 0.55, 0]
    landmarks[MIDDLE_PIP] = [0.50, 0.40, 0]
    landmarks[MIDDLE_DIP] = [0.50, 0.30, 0]
    landmarks[MIDDLE_TIP] = [0.50, 0.20, 0]
    
    # Ring finger chain
    landmarks[RING_MCP] = [0.60, 0.58, 0]
    landmarks[RING_PIP] = [0.60, 0.45, 0]
    landmarks[RING_DIP] = [0.60, 0.35, 0]
    landmarks[RING_TIP] = [0.60, 0.25, 0]
    
    # Pinky finger chain
    landmarks[PINKY_MCP] = [0.68, 0.62, 0]
    landmarks[PINKY_PIP] = [0.68, 0.52, 0]
    landmarks[PINKY_DIP] = [0.68, 0.44, 0]
    landmarks[PINKY_TIP] = [0.68, 0.36, 0]
    
    return landmarks


def curl_finger(landmarks, finger, amount=0.8):
    """Curl a finger towards the palm"""
    if finger == 'thumb':
        base_y = landmarks[THUMB_CMC][1]
        landmarks[THUMB_TIP] = [0.38, base_y - 0.05, 0]
        landmarks[THUMB_IP] = [0.35, base_y - 0.02, 0]
    elif finger == 'index':
        base_y = landmarks[INDEX_MCP][1]
        landmarks[INDEX_TIP] = [0.42, base_y + 0.05, 0]
        landmarks[INDEX_DIP] = [0.41, base_y + 0.02, 0]
        landmarks[INDEX_PIP] = [0.40, base_y - 0.02, 0]
    elif finger == 'middle':
        base_y = landmarks[MIDDLE_MCP][1]
        landmarks[MIDDLE_TIP] = [0.52, base_y + 0.05, 0]
        landmarks[MIDDLE_DIP] = [0.51, base_y + 0.02, 0]
        landmarks[MIDDLE_PIP] = [0.50, base_y - 0.02, 0]
    elif finger == 'ring':
        base_y = landmarks[RING_MCP][1]
        landmarks[RING_TIP] = [0.62, base_y + 0.05, 0]
        landmarks[RING_DIP] = [0.61, base_y + 0.02, 0]
        landmarks[RING_PIP] = [0.60, base_y - 0.02, 0]
    elif finger == 'pinky':
        base_y = landmarks[PINKY_MCP][1]
        landmarks[PINKY_TIP] = [0.70, base_y + 0.05, 0]
        landmarks[PINKY_DIP] = [0.69, base_y + 0.02, 0]
        landmarks[PINKY_PIP] = [0.68, base_y - 0.02, 0]
    return landmarks


def extend_finger(landmarks, finger):
    """Extend a finger upward"""
    if finger == 'thumb':
        landmarks[THUMB_TIP] = [0.22, 0.50, 0]
        landmarks[THUMB_IP] = [0.25, 0.58, 0]
        landmarks[THUMB_MCP] = [0.28, 0.65, 0]
    elif finger == 'index':
        landmarks[INDEX_TIP] = [0.40, 0.18, 0]
        landmarks[INDEX_DIP] = [0.40, 0.28, 0]
        landmarks[INDEX_PIP] = [0.40, 0.40, 0]
    elif finger == 'middle':
        landmarks[MIDDLE_TIP] = [0.50, 0.15, 0]
        landmarks[MIDDLE_DIP] = [0.50, 0.25, 0]
        landmarks[MIDDLE_PIP] = [0.50, 0.38, 0]
    elif finger == 'ring':
        landmarks[RING_TIP] = [0.60, 0.18, 0]
        landmarks[RING_DIP] = [0.60, 0.28, 0]
        landmarks[RING_PIP] = [0.60, 0.40, 0]
    elif finger == 'pinky':
        landmarks[PINKY_TIP] = [0.68, 0.25, 0]
        landmarks[PINKY_DIP] = [0.68, 0.35, 0]
        landmarks[PINKY_PIP] = [0.68, 0.45, 0]
    return landmarks


def generate_letter_A(num_samples=200):
    """A: Fist with thumb on the side (thumb alongside index, not over)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl all fingers into fist
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Thumb alongside the fist (not over fingers)
        lm[THUMB_TIP] = [0.32, 0.55, 0]
        lm[THUMB_IP] = [0.30, 0.62, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_B(num_samples=200):
    """B: Flat hand, all 4 fingers up together, thumb tucked across palm"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # All fingers extended up
        lm = extend_finger(lm, 'index')
        lm = extend_finger(lm, 'middle')
        lm = extend_finger(lm, 'ring')
        lm = extend_finger(lm, 'pinky')
        # Thumb tucked across palm
        lm[THUMB_TIP] = [0.45, 0.65, 0]
        lm[THUMB_IP] = [0.38, 0.68, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_C(num_samples=200):
    """C: Curved hand like holding a cup - all fingers curved in C shape"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # All fingers curved forming C
        # Thumb curved
        lm[THUMB_TIP] = [0.30, 0.45, 0]
        lm[THUMB_IP] = [0.28, 0.55, 0]
        # Index curved
        lm[INDEX_TIP] = [0.42, 0.32, 0]
        lm[INDEX_DIP] = [0.45, 0.38, 0]
        lm[INDEX_PIP] = [0.44, 0.48, 0]
        # Middle curved
        lm[MIDDLE_TIP] = [0.52, 0.30, 0]
        lm[MIDDLE_DIP] = [0.55, 0.35, 0]
        lm[MIDDLE_PIP] = [0.54, 0.45, 0]
        # Ring curved
        lm[RING_TIP] = [0.60, 0.32, 0]
        lm[RING_DIP] = [0.62, 0.38, 0]
        lm[RING_PIP] = [0.61, 0.48, 0]
        # Pinky curved
        lm[PINKY_TIP] = [0.66, 0.38, 0]
        lm[PINKY_DIP] = [0.68, 0.44, 0]
        lm[PINKY_PIP] = [0.67, 0.52, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_D(num_samples=200):
    """D: Index finger up, others make circle touching thumb"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Index extended UP
        lm = extend_finger(lm, 'index')
        # Middle, ring, pinky curled touching thumb
        lm[MIDDLE_TIP] = [0.42, 0.58, 0]
        lm[RING_TIP] = [0.44, 0.60, 0]
        lm[PINKY_TIP] = [0.46, 0.62, 0]
        # Thumb touching other fingers
        lm[THUMB_TIP] = [0.40, 0.58, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_E(num_samples=200):
    """E: All fingers curled down, tips near palm, thumb tucked"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # All 4 fingers curled tightly
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Thumb tucked under fingers
        lm[THUMB_TIP] = [0.42, 0.68, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_F(num_samples=200):
    """F: Index and thumb make circle (OK sign), middle/ring/pinky UP"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Middle, ring, pinky extended UP
        lm = extend_finger(lm, 'middle')
        lm = extend_finger(lm, 'ring')
        lm = extend_finger(lm, 'pinky')
        # Index and thumb touching in circle
        lm[INDEX_TIP] = [0.35, 0.52, 0]
        lm[INDEX_DIP] = [0.38, 0.48, 0]
        lm[THUMB_TIP] = [0.33, 0.52, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_G(num_samples=200):
    """G: Index and thumb pointing SIDEWAYS (horizontal)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl middle, ring, pinky
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index pointing sideways (horizontal)
        lm[INDEX_TIP] = [0.20, 0.50, 0]
        lm[INDEX_DIP] = [0.28, 0.50, 0]
        lm[INDEX_PIP] = [0.35, 0.52, 0]
        # Thumb also pointing sideways
        lm[THUMB_TIP] = [0.20, 0.58, 0]
        lm[THUMB_IP] = [0.26, 0.60, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_H(num_samples=200):
    """H: Index and middle pointing SIDEWAYS (horizontal)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl ring, pinky
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index and middle pointing sideways
        lm[INDEX_TIP] = [0.20, 0.48, 0]
        lm[INDEX_DIP] = [0.28, 0.48, 0]
        lm[INDEX_PIP] = [0.35, 0.50, 0]
        lm[MIDDLE_TIP] = [0.20, 0.55, 0]
        lm[MIDDLE_DIP] = [0.28, 0.55, 0]
        lm[MIDDLE_PIP] = [0.35, 0.56, 0]
        # Thumb curled
        lm = curl_finger(lm, 'thumb')
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_I(num_samples=200):
    """I: Pinky UP only, fist closed"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl all except pinky
        lm = curl_finger(lm, 'thumb')
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        # Pinky extended UP
        lm = extend_finger(lm, 'pinky')
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_J(num_samples=200):
    """J: Like I (pinky up) but tilted/rotated for J motion"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl all except pinky
        lm = curl_finger(lm, 'thumb')
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        # Pinky extended but tilted (J motion endpoint)
        lm[PINKY_TIP] = [0.72, 0.30, 0]
        lm[PINKY_DIP] = [0.70, 0.40, 0]
        lm[PINKY_PIP] = [0.68, 0.50, 0]
        # Slightly rotated hand
        lm[WRIST] = [0.52, 0.85, 0]
        samples.append(lm + np.random.normal(0, 0.02, 63))
    return samples


def generate_letter_K(num_samples=200):
    """K: Index and middle UP in V, thumb BETWEEN them"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl ring, pinky
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index and middle up, spread apart
        lm[INDEX_TIP] = [0.35, 0.18, 0]
        lm[INDEX_DIP] = [0.36, 0.30, 0]
        lm[INDEX_PIP] = [0.38, 0.42, 0]
        lm[MIDDLE_TIP] = [0.55, 0.18, 0]
        lm[MIDDLE_DIP] = [0.54, 0.30, 0]
        lm[MIDDLE_PIP] = [0.52, 0.42, 0]
        # Thumb between index and middle
        lm[THUMB_TIP] = [0.45, 0.38, 0]
        lm[THUMB_IP] = [0.42, 0.48, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_L(num_samples=200):
    """L: L shape - index UP, thumb OUT perpendicular"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl middle, ring, pinky
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index UP
        lm = extend_finger(lm, 'index')
        # Thumb OUT to side (perpendicular)
        lm[THUMB_TIP] = [0.18, 0.62, 0]
        lm[THUMB_IP] = [0.24, 0.65, 0]
        lm[THUMB_MCP] = [0.30, 0.68, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_M(num_samples=200):
    """M: Thumb under 3 fingers (index, middle, ring over thumb)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # All fingers curled over thumb
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Thumb tucked UNDER the 3 fingers
        lm[THUMB_TIP] = [0.55, 0.72, 0]
        lm[THUMB_IP] = [0.48, 0.72, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_N(num_samples=200):
    """N: Thumb under 2 fingers (index, middle over thumb)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Fingers curled
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Thumb between index and middle
        lm[THUMB_TIP] = [0.48, 0.70, 0]
        lm[THUMB_IP] = [0.42, 0.72, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_O(num_samples=200):
    """O: All fingertips touch thumb making O/circle shape"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # All fingertips converge to touch thumb - making O
        center_x, center_y = 0.45, 0.48
        # Thumb tip at center
        lm[THUMB_TIP] = [center_x - 0.05, center_y, 0]
        lm[THUMB_IP] = [center_x - 0.08, center_y + 0.08, 0]
        # Index curved to center
        lm[INDEX_TIP] = [center_x, center_y - 0.03, 0]
        lm[INDEX_DIP] = [center_x + 0.02, center_y + 0.05, 0]
        # Middle curved to center
        lm[MIDDLE_TIP] = [center_x + 0.03, center_y, 0]
        lm[MIDDLE_DIP] = [center_x + 0.06, center_y + 0.08, 0]
        # Ring curved to center
        lm[RING_TIP] = [center_x + 0.05, center_y + 0.02, 0]
        lm[RING_DIP] = [center_x + 0.08, center_y + 0.10, 0]
        # Pinky curved to center
        lm[PINKY_TIP] = [center_x + 0.06, center_y + 0.05, 0]
        lm[PINKY_DIP] = [center_x + 0.09, center_y + 0.12, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_P(num_samples=200):
    """P: Like K but pointing DOWN"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl ring, pinky
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index and middle pointing DOWN
        lm[INDEX_TIP] = [0.35, 0.78, 0]
        lm[INDEX_DIP] = [0.36, 0.70, 0]
        lm[INDEX_PIP] = [0.38, 0.62, 0]
        lm[MIDDLE_TIP] = [0.50, 0.78, 0]
        lm[MIDDLE_DIP] = [0.50, 0.70, 0]
        lm[MIDDLE_PIP] = [0.50, 0.62, 0]
        # Thumb between
        lm[THUMB_TIP] = [0.42, 0.65, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_Q(num_samples=200):
    """Q: Like G but pointing DOWN"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl middle, ring, pinky
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index and thumb pointing DOWN
        lm[INDEX_TIP] = [0.40, 0.82, 0]
        lm[INDEX_DIP] = [0.40, 0.75, 0]
        lm[INDEX_PIP] = [0.40, 0.65, 0]
        lm[THUMB_TIP] = [0.35, 0.80, 0]
        lm[THUMB_IP] = [0.34, 0.72, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_R(num_samples=200):
    """R: Index and middle crossed (middle over index)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl ring, pinky, thumb
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        lm = curl_finger(lm, 'thumb')
        # Index and middle up but CROSSED
        lm[INDEX_TIP] = [0.46, 0.18, 0]
        lm[INDEX_DIP] = [0.44, 0.28, 0]
        lm[INDEX_PIP] = [0.42, 0.40, 0]
        lm[MIDDLE_TIP] = [0.42, 0.18, 0]  # Middle crosses over index
        lm[MIDDLE_DIP] = [0.46, 0.28, 0]
        lm[MIDDLE_PIP] = [0.48, 0.40, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_S(num_samples=200):
    """S: Fist with thumb OVER fingers (closed fist)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # All fingers curled tight
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Thumb OVER the curled fingers
        lm[THUMB_TIP] = [0.45, 0.58, 0]
        lm[THUMB_IP] = [0.40, 0.60, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_T(num_samples=200):
    """T: Thumb between index and middle, others curled"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # All fingers curled
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Thumb pokes out between index and middle
        lm[THUMB_TIP] = [0.45, 0.52, 0]
        lm[THUMB_IP] = [0.42, 0.58, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_U(num_samples=200):
    """U: Index and middle UP together (parallel, not spread)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl ring, pinky, thumb
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        lm = curl_finger(lm, 'thumb')
        # Index and middle up TOGETHER (parallel)
        lm[INDEX_TIP] = [0.43, 0.18, 0]
        lm[INDEX_DIP] = [0.43, 0.28, 0]
        lm[INDEX_PIP] = [0.42, 0.40, 0]
        lm[MIDDLE_TIP] = [0.50, 0.18, 0]
        lm[MIDDLE_DIP] = [0.50, 0.28, 0]
        lm[MIDDLE_PIP] = [0.50, 0.40, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_V(num_samples=200):
    """V: Peace sign - index and middle UP and SPREAD apart"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl ring, pinky, thumb
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        lm = curl_finger(lm, 'thumb')
        # Index and middle up and SPREAD
        lm[INDEX_TIP] = [0.35, 0.18, 0]
        lm[INDEX_DIP] = [0.37, 0.28, 0]
        lm[INDEX_PIP] = [0.40, 0.40, 0]
        lm[MIDDLE_TIP] = [0.58, 0.18, 0]
        lm[MIDDLE_DIP] = [0.56, 0.28, 0]
        lm[MIDDLE_PIP] = [0.52, 0.40, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_W(num_samples=200):
    """W: Index, middle, ring UP and spread (3 fingers)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl pinky, thumb
        lm = curl_finger(lm, 'pinky')
        lm = curl_finger(lm, 'thumb')
        # Three fingers up and spread
        lm[INDEX_TIP] = [0.32, 0.18, 0]
        lm[INDEX_DIP] = [0.34, 0.28, 0]
        lm[INDEX_PIP] = [0.38, 0.42, 0]
        lm[MIDDLE_TIP] = [0.48, 0.15, 0]
        lm[MIDDLE_DIP] = [0.48, 0.26, 0]
        lm[MIDDLE_PIP] = [0.48, 0.40, 0]
        lm[RING_TIP] = [0.62, 0.18, 0]
        lm[RING_DIP] = [0.60, 0.28, 0]
        lm[RING_PIP] = [0.58, 0.42, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_X(num_samples=200):
    """X: Index finger bent/hooked (like a hook)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl all except index
        lm = curl_finger(lm, 'thumb')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index bent/hooked
        lm[INDEX_TIP] = [0.42, 0.42, 0]
        lm[INDEX_DIP] = [0.38, 0.38, 0]
        lm[INDEX_PIP] = [0.40, 0.48, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_Y(num_samples=200):
    """Y: Thumb and pinky OUT (hang loose sign)"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl index, middle, ring
        lm = curl_finger(lm, 'index')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        # Thumb OUT to side
        lm[THUMB_TIP] = [0.18, 0.58, 0]
        lm[THUMB_IP] = [0.24, 0.62, 0]
        # Pinky OUT/UP
        lm = extend_finger(lm, 'pinky')
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def generate_letter_Z(num_samples=200):
    """Z: Index pointing (draws Z in air) - static position"""
    samples = []
    for _ in range(num_samples):
        lm = base_hand()
        # Curl all except index
        lm = curl_finger(lm, 'thumb')
        lm = curl_finger(lm, 'middle')
        lm = curl_finger(lm, 'ring')
        lm = curl_finger(lm, 'pinky')
        # Index pointing out
        lm[INDEX_TIP] = [0.32, 0.35, 0]
        lm[INDEX_DIP] = [0.36, 0.42, 0]
        lm[INDEX_PIP] = [0.40, 0.50, 0]
        samples.append(lm + np.random.normal(0, 0.015, 63))
    return samples


def main():
    print("=" * 60)
    print("🤟 ASL Alphabet Model Training (Accurate Version)")
    print("=" * 60)
    
    generators = {
        'A': generate_letter_A,
        'B': generate_letter_B,
        'C': generate_letter_C,
        'D': generate_letter_D,
        'E': generate_letter_E,
        'F': generate_letter_F,
        'G': generate_letter_G,
        'H': generate_letter_H,
        'I': generate_letter_I,
        'J': generate_letter_J,
        'K': generate_letter_K,
        'L': generate_letter_L,
        'M': generate_letter_M,
        'N': generate_letter_N,
        'O': generate_letter_O,
        'P': generate_letter_P,
        'Q': generate_letter_Q,
        'R': generate_letter_R,
        'S': generate_letter_S,
        'T': generate_letter_T,
        'U': generate_letter_U,
        'V': generate_letter_V,
        'W': generate_letter_W,
        'X': generate_letter_X,
        'Y': generate_letter_Y,
        'Z': generate_letter_Z,
    }
    
    X_train = []
    y_train = []
    
    print("\n📊 Generating accurate training data...")
    for letter, generator in generators.items():
        samples = generator(num_samples=200)
        X_train.extend(samples)
        y_train.extend([letter] * len(samples))
        print(f"  {letter}: {len(samples)} samples")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\n✓ Total: {len(X_train)} samples, 26 classes")
    
    # Train model
    print("\n⏳ Training neural network...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=500,
        early_stopping=False,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_scaled, y_train)
    
    accuracy = model.score(X_scaled, y_train)
    print(f"\n✅ Training accuracy: {accuracy*100:.1f}%")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'classes': list(generators.keys())
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'sign_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved to {model_path}")
    print("\n🎉 Done! Restart the backend to use the trained model.")


if __name__ == '__main__':
    main()
