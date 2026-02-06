"""
Pre-train ASL Alphabet Model with Synthetic Landmark Data
Based on actual ASL fingerspelling hand positions
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ASL Alphabet landmark patterns (normalized finger positions)
# Each letter has characteristic finger states: 0=closed, 1=open, 0.5=bent
# Format: [thumb, index, middle, ring, pinky] openness + angles

def generate_landmarks_for_letter(letter, num_samples=100):
    """Generate synthetic hand landmarks for each ASL letter"""
    samples = []
    
    # Base hand structure: 21 landmarks x 3 (x, y, z)
    # Key landmarks: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
    
    for _ in range(num_samples):
        # Start with base hand position
        landmarks = np.zeros(63)  # 21 * 3
        
        # Add noise for variation
        noise = np.random.normal(0, 0.02, 63)
        
        # Wrist at center bottom
        landmarks[0:3] = [0.5, 0.8, 0]
        
        # Define finger positions based on letter
        if letter == 'A':  # Fist, thumb beside
            landmarks = make_fist(landmarks)
            landmarks[12] = 0.45  # Thumb beside fist
            
        elif letter == 'B':  # Flat hand, fingers up, thumb tucked
            landmarks = make_flat_hand(landmarks)
            landmarks[9:12] = [0.4, 0.7, 0]  # Thumb tucked
            
        elif letter == 'C':  # Curved like holding cup
            landmarks = make_c_shape(landmarks)
            
        elif letter == 'D':  # Index up, others make circle with thumb
            landmarks = make_d_shape(landmarks)
            
        elif letter == 'E':  # Fingers curled, thumb across
            landmarks = make_e_shape(landmarks)
            
        elif letter == 'F':  # OK sign, 3 fingers up
            landmarks = make_f_shape(landmarks)
            
        elif letter == 'G':  # Index and thumb pointing sideways
            landmarks = make_g_shape(landmarks)
            
        elif letter == 'H':  # Index and middle pointing sideways
            landmarks = make_h_shape(landmarks)
            
        elif letter == 'I':  # Pinky up only, fist closed
            landmarks = make_fist(landmarks)
            # Pinky tip UP (landmark 20 = indices 60:63)
            landmarks[60:63] = [0.68, 0.2, 0]  # Pinky tip extended up
            landmarks[51:54] = [0.62, 0.58, 0]  # Pinky MCP
            
        elif letter == 'J':  # Like I but hand tilted, pinky draws J
            landmarks = make_fist(landmarks)
            # Pinky tip to the side/tilted (landmark 20)
            landmarks[60:63] = [0.72, 0.28, 0]  # Pinky tip extended, tilted
            landmarks[51:54] = [0.64, 0.56, 0]  # Pinky MCP
            # Slightly rotated wrist
            landmarks[0:3] = [0.52, 0.78, 0]
            
        elif letter == 'K':  # Index and middle up in V, thumb between
            landmarks = make_k_shape(landmarks)
            
        elif letter == 'L':  # L shape - index up, thumb out
            landmarks = make_l_shape(landmarks)
            
        elif letter == 'M':  # Thumb under 3 fingers
            landmarks = make_m_shape(landmarks)
            
        elif letter == 'N':  # Thumb under 2 fingers  
            landmarks = make_n_shape(landmarks)
            
        elif letter == 'O':  # Fingers touch thumb in O
            landmarks = make_o_shape(landmarks)
            
        elif letter == 'P':  # Like K pointing down
            landmarks = make_p_shape(landmarks)
            
        elif letter == 'Q':  # Like G pointing down
            landmarks = make_q_shape(landmarks)
            
        elif letter == 'R':  # Crossed fingers
            landmarks = make_r_shape(landmarks)
            
        elif letter == 'S':  # Fist, thumb over fingers
            landmarks = make_fist(landmarks)
            landmarks[9:12] = [0.5, 0.55, 0]  # Thumb over
            
        elif letter == 'T':  # Thumb between index and middle
            landmarks = make_t_shape(landmarks)
            
        elif letter == 'U':  # Index and middle up together
            landmarks = make_u_shape(landmarks)
            
        elif letter == 'V':  # Peace sign
            landmarks = make_v_shape(landmarks)
            
        elif letter == 'W':  # 3 fingers up spread
            landmarks = make_w_shape(landmarks)
            
        elif letter == 'X':  # Index bent like hook
            landmarks = make_x_shape(landmarks)
            
        elif letter == 'Y':  # Thumb and pinky out (hang loose)
            landmarks = make_y_shape(landmarks)
            
        elif letter == 'Z':  # Index draws Z (static: index pointing)
            landmarks = make_fist(landmarks)
            landmarks[21:24] = [0.6, 0.3, 0]  # Index out
        
        landmarks += noise
        samples.append(landmarks)
    
    return np.array(samples)

def make_fist(landmarks):
    """Base fist position - all fingers curled"""
    # Wrist
    landmarks[0:3] = [0.5, 0.8, 0]
    
    # Thumb folded across palm
    landmarks[12:15] = [0.4, 0.58, 0]  # Thumb tip (landmark 4)
    
    # All fingertips curled down (tips near palm, below MCP)
    landmarks[24:27] = [0.42, 0.58, 0]  # Index tip (landmark 8)
    landmarks[36:39] = [0.48, 0.56, 0]  # Middle tip (landmark 12)
    landmarks[48:51] = [0.54, 0.58, 0]  # Ring tip (landmark 16)
    landmarks[60:63] = [0.60, 0.60, 0]  # Pinky tip (landmark 20)
    
    # MCP joints
    landmarks[15:18] = [0.38, 0.52, 0]  # Index MCP
    landmarks[27:30] = [0.48, 0.50, 0]  # Middle MCP
    landmarks[39:42] = [0.56, 0.52, 0]  # Ring MCP
    landmarks[51:54] = [0.62, 0.55, 0]  # Pinky MCP
    
    return landmarks

def make_flat_hand(landmarks):
    """Flat hand, all fingers extended up - used for B"""
    landmarks[0:3] = [0.5, 0.85, 0]  # Wrist
    
    # Thumb tucked to side
    landmarks[12:15] = [0.32, 0.55, 0]  # Thumb tip
    
    # All fingertips UP (low Y value = high position)
    landmarks[24:27] = [0.38, 0.15, 0]  # Index tip
    landmarks[36:39] = [0.48, 0.12, 0]  # Middle tip
    landmarks[48:51] = [0.58, 0.15, 0]  # Ring tip
    landmarks[60:63] = [0.68, 0.20, 0]  # Pinky tip
    
    # MCP joints
    landmarks[15:18] = [0.38, 0.50, 0]  # Index MCP
    landmarks[27:30] = [0.48, 0.48, 0]  # Middle MCP
    landmarks[39:42] = [0.58, 0.50, 0]  # Ring MCP
    landmarks[51:54] = [0.66, 0.52, 0]  # Pinky MCP
    
    return landmarks

def make_c_shape(landmarks):
    """C shape - curved hand like holding a cup
    Key feature: fingers curved but NOT touching thumb, form a C
    """
    landmarks[0:3] = [0.5, 0.8, 0]  # Wrist
    
    # Thumb curved inward but separate from fingers
    landmarks[12:15] = [0.32, 0.48, 0]  # Thumb tip
    
    # Fingertips curved but spread out more than O
    landmarks[24:27] = [0.38, 0.35, 0]  # Index tip - curved
    landmarks[36:39] = [0.45, 0.32, 0]  # Middle tip - curved
    landmarks[48:51] = [0.52, 0.35, 0]  # Ring tip - curved
    landmarks[60:63] = [0.58, 0.40, 0]  # Pinky tip - curved
    
    # MCP joints
    landmarks[15:18] = [0.35, 0.55, 0]  # Index MCP
    landmarks[27:30] = [0.45, 0.52, 0]  # Middle MCP
    landmarks[39:42] = [0.55, 0.55, 0]  # Ring MCP
    landmarks[51:54] = [0.62, 0.58, 0]  # Pinky MCP
    
    return landmarks

def make_d_shape(landmarks):
    """D - index up, others circle with thumb"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.45, 0.5, 0]  # Thumb touching middle
    landmarks[21:24] = [0.5, 0.15, 0]  # Index UP
    landmarks[33:36] = [0.48, 0.5, 0]  # Middle touching thumb
    landmarks[45:48] = [0.52, 0.55, 0]
    landmarks[57:60] = [0.58, 0.6, 0]
    return landmarks

def make_e_shape(landmarks):
    """E - fingers curled, thumb across"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.4, 0.45, 0]
    landmarks[21:24] = [0.42, 0.48, 0]
    landmarks[33:36] = [0.48, 0.45, 0]
    landmarks[45:48] = [0.54, 0.48, 0]
    landmarks[57:60] = [0.6, 0.52, 0]
    return landmarks

def make_f_shape(landmarks):
    """F - OK sign, index+thumb circle, 3 up"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.38, 0.5, 0]   # Thumb
    landmarks[21:24] = [0.4, 0.48, 0]   # Index touching thumb
    landmarks[33:36] = [0.5, 0.15, 0]   # Middle UP
    landmarks[45:48] = [0.58, 0.18, 0]  # Ring UP
    landmarks[57:60] = [0.66, 0.22, 0]  # Pinky UP
    return landmarks

def make_g_shape(landmarks):
    """G - index and thumb pointing sideways"""
    landmarks[0:3] = [0.5, 0.7, 0]
    landmarks[9:12] = [0.25, 0.55, 0]  # Thumb pointing side
    landmarks[21:24] = [0.25, 0.5, 0]   # Index pointing side
    landmarks[33:36] = [0.5, 0.6, 0]
    landmarks[45:48] = [0.55, 0.62, 0]
    landmarks[57:60] = [0.6, 0.65, 0]
    return landmarks

def make_h_shape(landmarks):
    """H - index and middle pointing sideways"""
    landmarks[0:3] = [0.5, 0.7, 0]
    landmarks[9:12] = [0.55, 0.6, 0]
    landmarks[21:24] = [0.2, 0.5, 0]   # Index sideways
    landmarks[33:36] = [0.22, 0.55, 0]  # Middle sideways
    landmarks[45:48] = [0.55, 0.62, 0]
    landmarks[57:60] = [0.6, 0.65, 0]
    return landmarks

def make_k_shape(landmarks):
    """K - V shape with thumb between"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.42, 0.4, 0]   # Thumb between
    landmarks[21:24] = [0.35, 0.15, 0]  # Index up
    landmarks[33:36] = [0.55, 0.15, 0]  # Middle up spread
    landmarks[45:48] = [0.58, 0.6, 0]
    landmarks[57:60] = [0.62, 0.65, 0]
    return landmarks

def make_l_shape(landmarks):
    """L - index up, thumb out perpendicular"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.25, 0.6, 0]   # Thumb OUT to side
    landmarks[21:24] = [0.5, 0.15, 0]  # Index UP
    landmarks[33:36] = [0.52, 0.6, 0]
    landmarks[45:48] = [0.56, 0.62, 0]
    landmarks[57:60] = [0.6, 0.65, 0]
    return landmarks

def make_m_shape(landmarks):
    """M - thumb under 3 fingers"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.52, 0.65, 0]  # Thumb under
    landmarks[21:24] = [0.4, 0.5, 0]
    landmarks[33:36] = [0.48, 0.48, 0]
    landmarks[45:48] = [0.56, 0.5, 0]
    landmarks[57:60] = [0.62, 0.58, 0]
    return landmarks

def make_n_shape(landmarks):
    """N - thumb under 2 fingers"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.5, 0.62, 0]
    landmarks[21:24] = [0.42, 0.5, 0]
    landmarks[33:36] = [0.5, 0.48, 0]
    landmarks[45:48] = [0.56, 0.58, 0]
    landmarks[57:60] = [0.6, 0.62, 0]
    return landmarks

def make_o_shape(landmarks):
    """O - all fingertips touch thumb in a circle
    Key feature: all fingertips clustered together near thumb tip
    """
    landmarks[0:3] = [0.5, 0.8, 0]  # Wrist
    
    # Thumb tip pointing inward
    landmarks[12:15] = [0.45, 0.42, 0]  # Thumb tip (landmark 4)
    
    # All fingertips clustered near thumb - close together!
    landmarks[24:27] = [0.44, 0.40, 0]  # Index tip (landmark 8)
    landmarks[36:39] = [0.46, 0.38, 0]  # Middle tip (landmark 12)
    landmarks[48:51] = [0.48, 0.40, 0]  # Ring tip (landmark 16)
    landmarks[60:63] = [0.50, 0.42, 0]  # Pinky tip (landmark 20)
    
    # Set MCP joints lower (higher Y) to show fingers are curled
    landmarks[15:18] = [0.35, 0.55, 0]  # Index MCP (landmark 5)
    landmarks[27:30] = [0.45, 0.52, 0]  # Middle MCP (landmark 9)
    landmarks[39:42] = [0.55, 0.55, 0]  # Ring MCP (landmark 13)
    landmarks[51:54] = [0.62, 0.58, 0]  # Pinky MCP (landmark 17)
    
    return landmarks

def make_p_shape(landmarks):
    """P - like K but pointing down"""
    landmarks[0:3] = [0.5, 0.5, 0]
    landmarks[9:12] = [0.42, 0.6, 0]
    landmarks[21:24] = [0.35, 0.85, 0]  # Index DOWN
    landmarks[33:36] = [0.55, 0.85, 0]  # Middle DOWN
    landmarks[45:48] = [0.58, 0.4, 0]
    landmarks[57:60] = [0.62, 0.45, 0]
    return landmarks

def make_q_shape(landmarks):
    """Q - like G but pointing down"""
    landmarks[0:3] = [0.5, 0.5, 0]
    landmarks[9:12] = [0.4, 0.8, 0]   # Thumb down
    landmarks[21:24] = [0.45, 0.85, 0]  # Index down
    landmarks[33:36] = [0.52, 0.45, 0]
    landmarks[45:48] = [0.56, 0.48, 0]
    landmarks[57:60] = [0.6, 0.52, 0]
    return landmarks

def make_r_shape(landmarks):
    """R - crossed index and middle"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.55, 0.6, 0]
    landmarks[21:24] = [0.48, 0.15, 0]  # Index up
    landmarks[33:36] = [0.45, 0.18, 0]  # Middle crossed over
    landmarks[45:48] = [0.56, 0.6, 0]
    landmarks[57:60] = [0.6, 0.62, 0]
    return landmarks

def make_t_shape(landmarks):
    """T - thumb between index and middle"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.46, 0.5, 0]  # Thumb between
    landmarks[21:24] = [0.44, 0.55, 0]
    landmarks[33:36] = [0.5, 0.52, 0]
    landmarks[45:48] = [0.55, 0.58, 0]
    landmarks[57:60] = [0.6, 0.62, 0]
    return landmarks

def make_u_shape(landmarks):
    """U - index and middle up together"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.55, 0.6, 0]
    landmarks[21:24] = [0.45, 0.15, 0]  # Index up
    landmarks[33:36] = [0.5, 0.15, 0]   # Middle up together
    landmarks[45:48] = [0.56, 0.6, 0]
    landmarks[57:60] = [0.6, 0.62, 0]
    return landmarks

def make_v_shape(landmarks):
    """V - peace sign, spread"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.55, 0.6, 0]
    landmarks[21:24] = [0.38, 0.15, 0]  # Index up spread
    landmarks[33:36] = [0.55, 0.15, 0]  # Middle up spread
    landmarks[45:48] = [0.56, 0.6, 0]
    landmarks[57:60] = [0.6, 0.62, 0]
    return landmarks

def make_w_shape(landmarks):
    """W - 3 fingers up spread"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.55, 0.6, 0]
    landmarks[21:24] = [0.35, 0.15, 0]  # Index up
    landmarks[33:36] = [0.48, 0.12, 0]  # Middle up
    landmarks[45:48] = [0.6, 0.15, 0]   # Ring up
    landmarks[57:60] = [0.62, 0.6, 0]
    return landmarks

def make_x_shape(landmarks):
    """X - index bent like hook"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.55, 0.6, 0]
    landmarks[21:24] = [0.45, 0.4, 0]  # Index bent/hooked
    landmarks[18:21] = [0.42, 0.35, 0]  # Index mid joint
    landmarks[33:36] = [0.52, 0.58, 0]
    landmarks[45:48] = [0.56, 0.6, 0]
    landmarks[57:60] = [0.6, 0.62, 0]
    return landmarks

def make_y_shape(landmarks):
    """Y - thumb and pinky out (hang loose)"""
    landmarks[0:3] = [0.5, 0.8, 0]
    landmarks[9:12] = [0.25, 0.55, 0]  # Thumb OUT
    landmarks[21:24] = [0.45, 0.58, 0]
    landmarks[33:36] = [0.5, 0.56, 0]
    landmarks[45:48] = [0.55, 0.58, 0]
    landmarks[57:60] = [0.72, 0.25, 0]  # Pinky OUT and up
    return landmarks


def main():
    print("="*50)
    print("🤟 Pre-training ASL Alphabet Model")
    print("="*50)
    
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    X_train = []
    y_train = []
    
    print("\n📊 Generating synthetic training data...")
    for letter in alphabet:
        samples = generate_landmarks_for_letter(letter, num_samples=150)
        X_train.extend(samples)
        y_train.extend([letter] * len(samples))
        print(f"  {letter}: {len(samples)} samples")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\n✓ Total: {len(X_train)} samples, {len(alphabet)} classes")
    
    # Train model
    print("\n⏳ Training neural network...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
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
        'classes': alphabet
    }
    
    with open('sign_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved to sign_model.pkl")
    print("\n🎉 Done! Restart the backend to use the trained model.")

if __name__ == '__main__':
    main()
