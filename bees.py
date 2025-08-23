import numpy as np
import cv2
import random
from typing import List, Set, Optional, Dict
from eye import CompoundEye
from bee_types import BeeBehaviorState, SpatialLocation, FoodSource

class ScoutBee:
    """
    Explorer bee that searches for new food sources (good frames).
    Contents:
        bee_id, state, compound_eye, current_location, memory_locations, exploration_radius, energy
    """
    
    def __init__(self, bee_id: int, compound_eye: CompoundEye):
        self.bee_id = bee_id
        self.state = BeeBehaviorState.EXPLORING
        self.compound_eye = compound_eye
        self.current_location: Optional[SpatialLocation] = None
        self.memory_locations: List[SpatialLocation] = []  # Memoria espacial
        self.exploration_radius = 5  # Radio de exploración en frames
        self.energy = 100.0  # Energía para exploración
        
    def explore_territory(self, frames: List[np.ndarray], 
                     explored_locations: Set[int]) -> Optional[FoodSource]:
        """
        Input: frames (List[np.ndarray]), explored_locations (Set[int])
        Context: Explores territory with more permissive criteria to find new food sources.
        Output: FoodSource object or None
        """
        
        if self.energy <= 0:
            return None
            
        # Selección bio-realista de área a explorar
        unexplored_indices = [i for i in range(len(frames)) 
                            if i not in explored_locations]
        
        if not unexplored_indices:
            return None
            
        # Exploración sesgada por memoria (aumentada probabilidad)
        if self.memory_locations and random.random() < 0.5:  # Aumentado de 0.3 a 0.5
            best_memory = max(self.memory_locations, key=lambda x: x.quality_estimate)
            target_idx = self._find_nearby_unexplored(best_memory.frame_index, 
                                                    unexplored_indices)
        else:
            target_idx = random.choice(unexplored_indices)
            
        # Evaluar frame objetivo
        frame_quality = self._evaluate_frame_for_sfm(frames[target_idx], 
                                                frames[max(0, target_idx-1)])
        
        # Crear ubicación espacial
        location = SpatialLocation(
            frame_index=target_idx,
            region_x=50,
            region_y=50,
            quality_estimate=frame_quality,
            explored=True
        )
        
        # Agregar a memoria
        self.memory_locations.append(location)
        if len(self.memory_locations) > 15:  # Aumentado de 10 a 15
            self.memory_locations.pop(0)
            
        # Consumir menos energía
        self.energy -= 3  # Reducido de 5 a 3
        
        # Umbral más permisivo para crear fuente
        if frame_quality > 0.15:  # Reducido de 0.3 a 0.15
            return FoodSource(
                location=location,
                nectar_amount=frame_quality,
                distance=self._calculate_processing_cost(frames[target_idx]),
                scouts_visited={self.bee_id}
            )
            
        return None
    
    def _emergency_exploration(self, frames: List[np.ndarray], 
                          start_idx: int, end_idx: int) -> List[int]:
        """
        Input: frames (List[np.ndarray]), start_idx (int), end_idx (int)
        Context: Emergency exploration to fill large gaps between selected frames.
        Output: List of frame indices (List[int])
        """
        emergency_frames = []
        gap_size = end_idx - start_idx
        
        # Número de frames a insertar basado en el tamaño del salto
        n_inserts = min(gap_size // 15, 5)  # Máximo 5 frames por salto
        
        if n_inserts <= 1:
            return emergency_frames
        
        # Dividir el salto en segmentos iguales
        segment_size = gap_size // (n_inserts + 1)
        
        for i in range(1, n_inserts + 1):
            candidate_idx = start_idx + (segment_size * i)
            
            # Evaluar frame candidato
            if candidate_idx < len(frames):
                frame_quality = self.scouts[0]._evaluate_frame_for_sfm(
                    frames[candidate_idx], 
                    frames[max(0, candidate_idx - 1)]
                )
                
                # Criterio muy permisivo para frames de emergencia
                if frame_quality > 0.1:  # Muy bajo umbral
                    emergency_frames.append(candidate_idx)
        
        return emergency_frames
    
    def _find_nearby_unexplored(self, center_idx: int, 
                              unexplored_indices: List[int]) -> int:
        """
        Input: center_idx (int), unexplored_indices (List[int])
        Context: Finds an unexplored frame near the center index.
        Output: Index of nearby unexplored frame (int)
        """
        nearby = [idx for idx in unexplored_indices 
                 if abs(idx - center_idx) <= self.exploration_radius]
        return random.choice(nearby) if nearby else random.choice(unexplored_indices)
    
    def _evaluate_frame_for_sfm(self, frame: np.ndarray, 
                           prev_frame: np.ndarray) -> float:
        """
        Input: frame (np.ndarray), prev_frame (np.ndarray)
        Context: Evaluates frame quality for SFM with more permissive criteria.
        Output: SFM quality score (float)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (umbral más bajo)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)  # Reducido de 1000 a 500
        
        # 2. Feature density (más permisivo)
        orb = cv2.ORB_create(nfeatures=300)  # Reducido de 500 a 300
        keypoints = orb.detect(gray, None)
        feature_density = len(keypoints) / (frame.shape[0] * frame.shape[1])
        feature_score = min(feature_density * 5000, 1.0)  # Reducido de 10000 a 5000
        
        # 3. Motion information (más sensible)
        motion_info = self.compound_eye.process_frame_motion(frame, prev_frame)
        motion_score = min(motion_info['motion_saliency'] / 25, 1.0)  # Reducido de 50 a 25
        
        # 4. Distribución espacial (más permisiva)
        if len(keypoints) > 5:  # Reducido de 10 a 5
            points = np.array([kp.pt for kp in keypoints])
            spatial_variance = np.var(points, axis=0).mean()
            spatial_score = min(spatial_variance / 5000, 1.0)  # Reducido de 10000 a 5000
        else:
            spatial_score = 0.2  # Score mínimo en lugar de 0.0
            
        # Combinación más balanceada
        sfm_quality = (
            sharpness_score * 0.25 +    # Reducido peso
            feature_score * 0.25 +      # Reducido peso
            motion_score * 0.25 +       # Aumentado peso
            spatial_score * 0.25        # Aumentado peso
        )
        
        # Bonus por tener al menos features básicas
        if len(keypoints) >= 3:
            sfm_quality += 0.1
        
        return min(sfm_quality, 1.0)
    
    def _calculate_processing_cost(self, frame: np.ndarray) -> float:
        """
        Input: frame (np.ndarray)
        Context: Calculates computational cost (distance in bee terms).
        Output: Cost value (float)
        """
        # Costo basado en resolución y complejidad
        pixel_count = frame.shape[0] * frame.shape[1]
        complexity = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                 cv2.CV_64F).var()
        
        return (pixel_count / 1000000) + (complexity / 10000)

class WorkerBee:
    """
    Worker bee that evaluates known food sources.
    Contents:
        bee_id, state, compound_eye, current_source, dance_intensity
    """
    
    def __init__(self, bee_id: int, compound_eye: CompoundEye):
        self.bee_id = bee_id
        self.state = BeeBehaviorState.EVALUATING
        self.compound_eye = compound_eye
        self.current_source: Optional[FoodSource] = None
        self.dance_intensity = 0.0
        
    def evaluate_food_source(self, source: FoodSource, 
                           frames: List[np.ndarray]) -> float:
        """
        Input: source (FoodSource), frames (List[np.ndarray])
        Context: Evaluates a food source (frame) in detail.
        Output: Integrated quality score (float)
        """
        
        frame_idx = source.location.frame_index
        if frame_idx >= len(frames):
            return 0.0
            
        frame = frames[frame_idx]
        prev_frame = frames[max(0, frame_idx - 1)]
        
        # Evaluación detallada usando compound eye
        motion_analysis = self.compound_eye.process_frame_motion(frame, prev_frame)
        
        # Análisis de correspondencias (crítico para SFM)
        correspondence_quality = self._analyze_correspondence_potential(frame, prev_frame)
        
        # Análisis de baseline (separación entre frames para triangulación)
        baseline_quality = self._analyze_baseline_potential(frame_idx, frames)
        
        # Score integrado para SFM
        integrated_quality = (
            source.nectar_amount * 0.4 +           # Calidad base
            correspondence_quality * 0.35 +        # Potencial de matching
            baseline_quality * 0.15 +              # Calidad de baseline
            motion_analysis['motion_saliency'] / 100 * 0.1  # Motion info
        )
        
        # Actualizar fuente
        source.nectar_amount = integrated_quality
        source.scouts_visited.add(self.bee_id)
        source.persistence += 1
        
        return integrated_quality
    
    def _analyze_correspondence_potential(self, frame: np.ndarray, 
                                        prev_frame: np.ndarray) -> float:
        """
        Input: frame (np.ndarray), prev_frame (np.ndarray)
        Context: Analyzes potential for finding correspondences between frames.
        Output: Correspondence quality score (float)
        """
        
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Detectar y describir features
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0
            
        # Matching de features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 8:  # Mínimo para matriz fundamental
            return 0.0
            
        # Calidad de matches
        good_matches = [m for m in matches if m.distance < 50]
        match_ratio = len(good_matches) / len(matches)
        
        # Distribución espacial de matches
        if len(good_matches) > 10:
            points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
            spatial_distribution = np.var(points1, axis=0).mean()
            spatial_score = min(spatial_distribution / 10000, 1.0)
        else:
            spatial_score = 0.0
            
        return match_ratio * 0.7 + spatial_score * 0.3
    
    def _analyze_baseline_potential(self, frame_idx: int, 
                                  frames: List[np.ndarray]) -> float:
        """
        Input: frame_idx (int), frames (List[np.ndarray])
        Context: Analyzes baseline quality for triangulation.
        Output: Baseline quality score (float)
        """
        
        # Evaluamos separación con frames anteriores y posteriores
        baseline_scores = []
        
        for offset in [-2, -1, 1, 2]:
            other_idx = frame_idx + offset
            if 0 <= other_idx < len(frames):
                # Separación temporal
                temporal_sep = abs(offset)
                
                # Diferencia visual (proxy para baseline geométrico)
                frame1 = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(frames[other_idx], cv2.COLOR_BGR2GRAY)
                
                visual_diff = np.mean(cv2.absdiff(frame1, frame2))
                
                # Score de baseline (balance entre separación y diferencia)
                baseline_score = (temporal_sep / 3.0) * 0.6 + (visual_diff / 255.0) * 0.4
                baseline_scores.append(min(baseline_score, 1.0))
                
        return np.mean(baseline_scores) if baseline_scores else 0.0

class WaggleDancer:
    """
    Bee that performs waggle dance to communicate high-quality locations.
    Contents:
        bee_id, state, dance_duration, followers
    """
    
    def __init__(self, bee_id: int):
        self.bee_id = bee_id
        self.state = BeeBehaviorState.DANCING
        self.dance_duration = 0
        self.followers: Set[int] = set()
        
    def perform_waggle_dance(self, food_source: FoodSource, 
                           current_position: int) -> Dict[str, float]:
        """
        Input: food_source (FoodSource), current_position (int)
        Context: Performs waggle dance, communicating information about the food source.
        Output: Dictionary with dance information
        """
        
        # Información comunicada en la danza (como abejas reales)
        distance_to_source = abs(food_source.location.frame_index - current_position)
        
        # Intensidad de danza proporcional a calidad/distancia
        dance_intensity = food_source.get_profitability()
        
        # Duración de danza (proporcional a calidad)
        dance_duration = min(dance_intensity * 10, 30)  # Max 30 segundos
        
        # Dirección (en términos de índices de frame)
        direction = 1 if food_source.location.frame_index > current_position else -1
        
        dance_info = {
            'intensity': dance_intensity,
            'duration': dance_duration,
            'distance': distance_to_source,
            'direction': direction,
            'nectar_quality': food_source.nectar_amount,
            'dancer_id': self.bee_id
        }
        
        self.dance_duration = dance_duration
        
        return dance_info
