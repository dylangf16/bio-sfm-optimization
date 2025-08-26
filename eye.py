import numpy as np
import cv2
from typing import Tuple, List, Dict

class MotionReceptor:
    """
    Simulates an individual photoreceptor in an ommatidium.
    Contents:
        position, sensitivity, previous_intensity, temporal_buffer
    """
    
    def __init__(self, position: Tuple[int, int], sensitivity: float = 1.0):
        self.position = position
        self.sensitivity = sensitivity
        self.previous_intensity = 0.0
        self.temporal_buffer = []  # Biological temporal memory
        self.tau = 0.033  # Temporal delay constant
        
    def detect_motion(self, current_intensity: float, dt: float = 0.033) -> float:
        """
        Input: current_intensity (float), dt (float)
        Context: Detects motion as a real photoreceptor using temporal difference.
        Output: Absolute motion signal (float)
        """
        # Diferencia temporal (como Elementary Motion Detectors - EMD)
        motion_signal = 0.0
        
        if len(self.temporal_buffer) > 0:
            # Reichardt correlation model
            intensity_diff = current_intensity - self.previous_intensity
            time_diff = max(dt, 0.001)
            
            # EMD response with temporal correlation
            motion_signal = (intensity_diff * self.sensitivity) / time_diff
            
        # Update temporal buffer (max 5 samples as in real insects)
        self.temporal_buffer.append(current_intensity)
        if len(self.temporal_buffer) > 5:
            self.temporal_buffer.pop(0)
            
        self.previous_intensity = current_intensity
        return abs(motion_signal)

class CompoundEye:
    """
    Real compound eye with multiple ommatidia.
    Contents:
        n_ommatidia, fov_degrees, ommatidia, motion_field
    """
    
    def __init__(self, n_ommatidia: int = 200, fov_degrees: float = 180):
        self.n_ommatidia = n_ommatidia
        self.fov_degrees = fov_degrees
        self.ommatidia = []
        self.motion_field = np.zeros((n_ommatidia,))
        
        self._initialize_ommatidia()
        
    def _initialize_ommatidia(self):
        """
        Input: None
        Context: Initializes ommatidia with a realistic hexagonal distribution.
        Output: None
        """
        for i in range(self.n_ommatidia):
            # Angular position: θ_i = (i/N_ommatidia) * FOV_degrees
            angle = (i / self.n_ommatidia) * self.fov_degrees
            
            # Each ommatidium has 8 receptors (R1-R8)
            receptors = []
            for j in range(8):
                pos_x = int(np.cos(np.radians(angle)) * 50 + 50)
                pos_y = int(np.sin(np.radians(angle)) * 50 + 50)
                # R1-R6 more sensitive than R7-R8
                sensitivity = 1.0 if j < 6 else 0.5
                
                receptors.append(MotionReceptor((pos_x, pos_y), sensitivity))
                
            self.ommatidia.append(receptors)
    
    def process_frame_motion(self, frame: np.ndarray, prev_frame: np.ndarray, 
                           dt: float = 0.033) -> Dict[str, float]:
        """
        Input: frame (np.ndarray), prev_frame (np.ndarray), dt (float)
        Context: Processes motion using a bio-realistic model.
        Output: Dictionary with global_motion, local_variations, directional_bias, motion_saliency
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        motion_responses = []
        
        # Process each ommatidium
        for omatidium_idx, receptors in enumerate(self.ommatidia):
            omatidium_response = 0.0
            
            for receptor in receptors:
                # Get intensity at receptor position
                x = min(max(0, receptor.position[0] * w // 100), w-1)
                y = min(max(0, receptor.position[1] * h // 100), h-1)
                
                current_intensity = float(gray[y, x])
                motion_response = receptor.detect_motion(current_intensity, dt)
                omatidium_response += motion_response
                
            # Average per ommatidium
            motion_responses.append(omatidium_response / len(receptors))
        
        global_motion = np.mean(motion_responses)
        local_variations = np.std(motion_responses)
        directional_bias = self._compute_directional_bias(motion_responses)
        
        return {
            'global_motion': global_motion,
            'local_variations': local_variations,
            'directional_bias': directional_bias,
            'motion_saliency': global_motion * (1 + local_variations)
        }
    
    def _compute_directional_bias(self, responses: List[float]) -> float:
        """
        Input: responses (List[float])
        Context: Calculates directional bias of motion.
        Output: Directional bias magnitude (float)
        """
        if len(responses) < 2:
            return 0.0
            
        angles = np.linspace(0, 2*np.pi, len(responses))
        x_component = np.sum(np.array(responses) * np.cos(angles))
        y_component = np.sum(np.array(responses) * np.sin(angles))
        
        magnitude = np.sqrt(x_component**2 + y_component**2)
        return magnitude / max(np.sum(responses), 0.001)
