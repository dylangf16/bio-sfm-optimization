import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import logging
from abc import ABC, abstractmethod
import random
from enum import Enum
import shutil


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BeeBehaviorState(Enum):
    EXPLORING = "exploring"
    EVALUATING = "evaluating" 
    DANCING = "dancing"
    FOLLOWING = "following"
    INACTIVE = "inactive"

@dataclass
class SpatialLocation:
    """Representa una ubicación en el espacio de frames para exploración"""
    frame_index: int
    region_x: int  # Región espacial dentro del frame
    region_y: int
    quality_estimate: float = 0.0
    explored: bool = False
    exploitation_count: int = 0

@dataclass
class FoodSource:
    """Fuente de alimento = Frame con alta calidad para SFM"""
    location: SpatialLocation
    nectar_amount: float  # Calidad SFM del frame
    distance: float       # Costo computacional
    persistence: int = 0  # Cuántos ciclos ha sido visitada
    scouts_visited: Set[int] = field(default_factory=set)
    
    def get_profitability(self) -> float:
        """Rentabilidad = calidad / costo (bio-realistic)"""
        return self.nectar_amount / max(self.distance, 0.1)

class MotionReceptor:
    """Simula un fotorreceptor individual en un omatidio"""
    
    def __init__(self, position: Tuple[int, int], sensitivity: float = 1.0):
        self.position = position
        self.sensitivity = sensitivity
        self.previous_intensity = 0.0
        self.temporal_buffer = []  # Historial temporal como en insectos reales
        
    def detect_motion(self, current_intensity: float, dt: float) -> float:
        """Detecta movimiento como un fotorreceptor real"""
        # Diferencia temporal (como Elementary Motion Detectors - EMD)
        motion_signal = 0.0
        
        if len(self.temporal_buffer) > 0:
            # Correlación temporal-espacial (modelo Reichardt)
            intensity_diff = current_intensity - self.previous_intensity
            time_diff = dt
            
            # EMD response: correlación entre intensidades vecinas
            motion_signal = intensity_diff * self.sensitivity / max(time_diff, 0.001)
            
        # Actualizar buffer temporal (max 5 muestras como en insectos)
        self.temporal_buffer.append(current_intensity)
        if len(self.temporal_buffer) > 5:
            self.temporal_buffer.pop(0)
            
        self.previous_intensity = current_intensity
        return abs(motion_signal)

class CompoundEye:
    """OJO COMPUESTO REAL con múltiples omatidios"""
    
    def __init__(self, n_ommatidia: int = 100, fov_degrees: float = 180):
        self.n_ommatidia = n_ommatidia
        self.fov_degrees = fov_degrees
        self.ommatidia = []
        self.motion_field = np.zeros((n_ommatidia,))
        
        # Crear receptores distribuidos como en ojo real
        self._initialize_ommatidia()
        
    def _initialize_ommatidia(self):
        """Inicializar omatidios con distribución realista"""
        # Distribución hexagonal como en ojos reales
        for i in range(self.n_ommatidia):
            # Posición angular (similar a Drosophila)
            angle = (i / self.n_ommatidia) * self.fov_degrees
            
            # Cada omatidio tiene múltiples receptores
            receptors = []
            for j in range(8):  # 8 receptores por omatidio (R1-R8)
                pos_x = int(np.cos(np.radians(angle)) * 50 + 50)
                pos_y = int(np.sin(np.radians(angle)) * 50 + 50)
                sensitivity = 1.0 if j < 6 else 0.5  # R1-R6 más sensibles
                
                receptors.append(MotionReceptor((pos_x, pos_y), sensitivity))
                
            self.ommatidia.append(receptors)
    
    def process_frame_motion(self, frame: np.ndarray, prev_frame: np.ndarray, 
                           dt: float = 0.033) -> Dict[str, float]:
        """Procesa movimiento con modelo bio-realista"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        motion_responses = []
        
        # Procesar cada omatidio
        for omatidium_idx, receptors in enumerate(self.ommatidia):
            omatidium_response = 0.0
            
            for receptor in receptors:
                # Obtener intensidad en la posición del receptor
                x = min(max(0, receptor.position[0] * w // 100), w-1)
                y = min(max(0, receptor.position[1] * h // 100), h-1)
                
                current_intensity = float(gray[y, x])
                motion_response = receptor.detect_motion(current_intensity, dt)
                omatidium_response += motion_response
                
            # Promedio por omatidio
            motion_responses.append(omatidium_response / len(receptors))
        
        # Integración global (como en lóbulo óptico real)
        global_motion = np.mean(motion_responses)
        local_variations = np.std(motion_responses)
        directional_bias = self._compute_directional_bias(motion_responses)
        
        return {
            'global_motion': global_motion,
            'local_variations': local_variations,
            'directional_bias': directional_bias,
            'motion_saliency': global_motion * (1 + local_variations)  # Combinación bio-inspirada
        }
    
    def _compute_directional_bias(self, responses: List[float]) -> float:
        """Calcula sesgo direccional del movimiento"""
        if len(responses) < 2:
            return 0.0
            
        # Crear vector de movimiento direccional
        angles = np.linspace(0, 2*np.pi, len(responses))
        x_component = np.sum(np.array(responses) * np.cos(angles))
        y_component = np.sum(np.array(responses) * np.sin(angles))
        
        # Magnitud del vector resultante normalizada
        magnitude = np.sqrt(x_component**2 + y_component**2)
        return magnitude / max(np.sum(responses), 0.001)

class ScoutBee:
    """Abeja exploradora que busca nuevas fuentes de alimento (frames buenos)"""
    
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
        """Explora territorio con criterios más permisivos"""
        
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
        """Exploración de emergencia para llenar saltos grandes"""
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
        """Encuentra frame no explorado cerca del centro"""
        nearby = [idx for idx in unexplored_indices 
                 if abs(idx - center_idx) <= self.exploration_radius]
        return random.choice(nearby) if nearby else random.choice(unexplored_indices)
    
    def _evaluate_frame_for_sfm(self, frame: np.ndarray, 
                           prev_frame: np.ndarray) -> float:
        """Evalúa calidad de frame para SFM con criterios más permisivos"""
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
        """Calcula costo computacional (distancia en términos de abejas)"""
        # Costo basado en resolución y complejidad
        pixel_count = frame.shape[0] * frame.shape[1]
        complexity = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                 cv2.CV_64F).var()
        
        return (pixel_count / 1000000) + (complexity / 10000)

class WorkerBee:
    """Abeja trabajadora que evalúa fuentes conocidas"""
    
    def __init__(self, bee_id: int, compound_eye: CompoundEye):
        self.bee_id = bee_id
        self.state = BeeBehaviorState.EVALUATING
        self.compound_eye = compound_eye
        self.current_source: Optional[FoodSource] = None
        self.dance_intensity = 0.0
        
    def evaluate_food_source(self, source: FoodSource, 
                           frames: List[np.ndarray]) -> float:
        """Evalúa fuente de alimento (frame) en detalle"""
        
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
        """Analiza potencial para encontrar correspondencias entre frames"""
        
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
        """Analiza calidad de baseline para triangulación"""
        
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
    """Abeja que realiza waggle dance para comunicar ubicaciones de calidad"""
    
    def __init__(self, bee_id: int):
        self.bee_id = bee_id
        self.state = BeeBehaviorState.DANCING
        self.dance_duration = 0
        self.followers: Set[int] = set()
        
    def perform_waggle_dance(self, food_source: FoodSource, 
                           current_position: int) -> Dict[str, float]:
        """Realiza waggle dance comunicando información de la fuente"""
        
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

class HiveMind:
    """Mente colectiva que coordina el comportamiento del enjambre"""
    
    def __init__(self, n_scouts: int = 20, n_workers: int = 40):
        self.n_scouts = n_scouts
        self.n_workers = n_workers
        
        # Crear compound eye compartido (simulando sistema visual común)
        self.compound_eye = CompoundEye(n_ommatidia=200)
        
        # Crear abejas
        self.scouts = [ScoutBee(i, self.compound_eye) for i in range(n_scouts)]
        self.workers = [WorkerBee(i + n_scouts, self.compound_eye) 
                       for i in range(n_workers)]
        self.dancers = [WaggleDancer(i + n_scouts + n_workers) 
                       for i in range(min(n_scouts, 10))]  # Subset de dancers
        
        # Estado global
        self.known_food_sources: List[FoodSource] = []
        self.explored_locations: Set[int] = set()
        self.dance_floor: List[Dict] = []  # Información de danzas activas
        self.hive_knowledge: Dict[str, float] = {}
        
        # Comunicación entre abejas
        self.message_queue = queue.Queue()
        self.lock = threading.Lock()
        
    def foraging_cycle(self, frames: List[np.ndarray], 
                  max_cycles: int = 12) -> List[int]:  # Aumentado de 10 a 12
        """Ejecuta ciclos de forrajeo con más iteraciones"""
        
        logger.info(f"Iniciando forrajeo con {len(frames)} frames")
        
        # Almacenar frames para usar en emergency exploration
        self.frames = frames
        
        for cycle in range(max_cycles):
            logger.info(f"Ciclo de forrajeo {cycle + 1}/{max_cycles}")
            
            # Fases originales...
            self._scout_exploration_phase(frames)
            self._worker_evaluation_phase(frames)
            self._dance_communication_phase()
            self._dance_following_phase(frames)
            
            # Evaluación de convergencia más relajada
            if cycle >= 5 and self._check_convergence():  # Mínimo 5 ciclos
                logger.info(f"Convergencia alcanzada en ciclo {cycle + 1}")
                break
                
        # Selección final con frames disponibles
        selected_frames = self._select_final_frames_with_frames(frames)
        
        logger.info(f"Forrajeo completado. Seleccionados {len(selected_frames)} frames")
        return selected_frames
    
    def _detect_large_gaps(self, selected_indices: List[int]) -> List[Tuple[int, int]]:
        """Detecta saltos grandes entre frames seleccionados"""
        gaps = []
        for i in range(len(selected_indices) - 1):
            current = selected_indices[i]
            next_frame = selected_indices[i + 1]
            gap_size = next_frame - current
            
            if gap_size >= 10:  # Salto considerable
                gaps.append((current, next_frame))
        
        return gaps

    # Nueva función que incluye frames para emergency exploration
    def _select_final_frames_with_frames(self, frames: List[np.ndarray]) -> List[int]:
        """Versión de _select_final_frames que incluye emergency exploration"""
        
        if not self.known_food_sources:
            return []
            
        # Umbral más permisivo
        quality_threshold = np.percentile([s.nectar_amount for s in self.known_food_sources], 35)
        good_sources = [s for s in self.known_food_sources 
                    if s.nectar_amount >= quality_threshold]
        
        # Si muy pocos frames cumplen, relajar más
        if len(good_sources) < max(len(self.known_food_sources) * 0.25, 5):
            quality_threshold = np.percentile([s.nectar_amount for s in self.known_food_sources], 15)
            good_sources = [s for s in self.known_food_sources 
                        if s.nectar_amount >= quality_threshold]
        
        good_sources.sort(key=lambda s: s.get_profitability(), reverse=True)
        
        # Selección inicial
        selected_indices = []
        min_separation = 2
        
        for source in good_sources:
            frame_idx = source.location.frame_index
            too_close = any(abs(frame_idx - sel_idx) < min_separation 
                        for sel_idx in selected_indices)
            
            if not too_close:
                selected_indices.append(frame_idx)
        
        selected_indices.sort()
        
        # Emergency exploration para saltos grandes
        gaps = self._detect_large_gaps(selected_indices)
        emergency_frames = []
        
        for start_gap, end_gap in gaps:
            gap_size = end_gap - start_gap
            n_inserts = min(gap_size // 12, 6)  # Más frames de emergencia
            
            if n_inserts >= 1:
                segment_size = gap_size // (n_inserts + 1)
                
                for i in range(1, n_inserts + 1):
                    candidate_idx = start_gap + (segment_size * i)
                    
                    if 0 <= candidate_idx < len(frames):
                        # Evaluación simple para emergencia
                        frame = frames[candidate_idx]
                        prev_frame = frames[max(0, candidate_idx - 1)]
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        orb = cv2.ORB_create(nfeatures=100)
                        keypoints = orb.detect(gray, None)
                        
                        # Criterio muy básico pero efectivo
                        if len(keypoints) >= 3:
                            emergency_frames.append(candidate_idx)
        
        # Combinar y finalizar
        all_selected = selected_indices + emergency_frames
        all_selected = sorted(list(set(all_selected)))
        
        logger.info(f"Selección final: {len(selected_indices)} originales + "
                    f"{len(emergency_frames)} emergencia = {len(all_selected)} total")
        
        return all_selected
    
    def _scout_exploration_phase(self, frames: List[np.ndarray]):
        """Fase de exploración distribuida por scouts"""
        
        with ThreadPoolExecutor(max_workers=self.n_scouts) as executor:
            futures = []
            
            for scout in self.scouts:
                if scout.energy > 0:
                    future = executor.submit(scout.explore_territory, 
                                           frames, self.explored_locations)
                    futures.append((scout, future))
            
            # Recoger resultados
            for scout, future in futures:
                try:
                    food_source = future.result(timeout=10)
                    if food_source:
                        with self.lock:
                            self.known_food_sources.append(food_source)
                            self.explored_locations.add(food_source.location.frame_index)
                except Exception as e:
                    logger.warning(f"Scout {scout.bee_id} falló: {e}")
    
    def _worker_evaluation_phase(self, frames: List[np.ndarray]):
        """Fase de evaluación detallada por workers"""
        
        if not self.known_food_sources:
            return
            
        # Asignar workers a fuentes conocidas
        sources_to_evaluate = self.known_food_sources.copy()
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for i, worker in enumerate(self.workers):
                if sources_to_evaluate:
                    source = sources_to_evaluate[i % len(sources_to_evaluate)]
                    future = executor.submit(worker.evaluate_food_source, 
                                           source, frames)
                    futures.append((worker, source, future))
            
            # Actualizar evaluaciones
            for worker, source, future in futures:
                try:
                    quality = future.result(timeout=15)
                    worker.current_source = source
                    # La fuente ya se actualiza dentro del método
                except Exception as e:
                    logger.warning(f"Worker {worker.bee_id} falló: {e}")
    
    def _dance_communication_phase(self):
        """Fase de comunicación por waggle dance"""
        
        # Seleccionar mejores fuentes para danzar
        if not self.known_food_sources:
            return
            
        # Ordenar por rentabilidad
        sorted_sources = sorted(self.known_food_sources, 
                              key=lambda s: s.get_profitability(), 
                              reverse=True)
        
        top_sources = sorted_sources[:len(self.dancers)]
        
        # Realizar danzas
        self.dance_floor.clear()
        
        for dancer, source in zip(self.dancers, top_sources):
            # Posición promedio del enjambre (simplificación)
            avg_position = int(np.mean(list(self.explored_locations))) if self.explored_locations else 0
            
            dance_info = dancer.perform_waggle_dance(source, avg_position)
            dance_info['source'] = source
            
            self.dance_floor.append(dance_info)
    
    def _dance_following_phase(self, frames: List[np.ndarray]):
        """Fase donde abejas siguen información de danzas"""
        
        if not self.dance_floor:
            return
            
        # Scouts siguen danzas prometedoras
        for scout in self.scouts[:5]:  # Solo algunos siguen
            if scout.energy > 20:  # Energía suficiente
                # Seleccionar danza a seguir (sesgado por intensidad)
                dance_weights = [d['intensity'] for d in self.dance_floor]
                if sum(dance_weights) > 0:
                    selected_dance = np.random.choice(
                        self.dance_floor,
                        p=np.array(dance_weights) / sum(dance_weights)
                    )
                    
                    # Explorar cerca de la ubicación danzada
                    target_idx = selected_dance['source'].location.frame_index
                    nearby_indices = [i for i in range(max(0, target_idx-3), 
                                                     min(len(frames), target_idx+4))
                                    if i not in self.explored_locations]
                    
                    if nearby_indices:
                        # Evaluar área cercana
                        for idx in nearby_indices[:2]:  # Máximo 2 frames cercanos
                            if scout.energy > 0:
                                # Crear nueva exploración dirigida
                                frame_quality = scout._evaluate_frame_for_sfm(
                                    frames[idx], frames[max(0, idx-1)]
                                )
                                
                                if frame_quality > 0.25:
                                    location = SpatialLocation(
                                        frame_index=idx,
                                        region_x=50, region_y=50,
                                        quality_estimate=frame_quality,
                                        explored=True
                                    )
                                    
                                    new_source = FoodSource(
                                        location=location,
                                        nectar_amount=frame_quality,
                                        distance=scout._calculate_processing_cost(frames[idx]),
                                        scouts_visited={scout.bee_id}
                                    )
                                    
                                    with self.lock:
                                        self.known_food_sources.append(new_source)
                                        self.explored_locations.add(idx)
                                    
                                    scout.energy -= 10
    
    # Modificar la función _check_convergence para ser menos estricta
    def _check_convergence(self) -> bool:
        """Verifica convergencia con criterios más relajados"""
        
        if len(self.known_food_sources) < 3:  # Reducido de 5 a 3
            return False
            
        # Convergencia basada en número de fuentes encontradas
        if len(self.known_food_sources) >= len(self.scouts) * 2:
            return True
        
        # Convergencia basada en estabilidad (más permisiva)
        recent_qualities = [s.nectar_amount for s in self.known_food_sources[-15:]]
        
        if len(recent_qualities) >= 8:  # Reducido de 5 a 8 pero más muestras
            quality_variance = np.var(recent_qualities)
            return quality_variance < 0.05  # Aumentado de 0.01 a 0.05
        
        return False
    
    def _select_final_frames(self) -> List[int]:
        """Selección final más permisiva con continuidad mejorada"""
        
        if not self.known_food_sources:
            return []
            
        # Umbral más permisivo (percentil 40 en lugar de 60)
        quality_threshold = np.percentile([s.nectar_amount for s in self.known_food_sources], 40)
        good_sources = [s for s in self.known_food_sources 
                    if s.nectar_amount >= quality_threshold]
        
        # Si muy pocos frames cumplen, relajar más el umbral
        if len(good_sources) < len(self.known_food_sources) * 0.3:
            quality_threshold = np.percentile([s.nectar_amount for s in self.known_food_sources], 20)
            good_sources = [s for s in self.known_food_sources 
                        if s.nectar_amount >= quality_threshold]
        
        # Ordenar por rentabilidad
        good_sources.sort(key=lambda s: s.get_profitability(), reverse=True)
        
        # Selección inicial con separación mínima reducida
        selected_indices = []
        min_separation = 2  # Reducido de 3 a 2
        
        for source in good_sources:
            frame_idx = source.location.frame_index
            
            # Verificar separación mínima
            too_close = any(abs(frame_idx - sel_idx) < min_separation 
                        for sel_idx in selected_indices)
            
            if not too_close:
                selected_indices.append(frame_idx)
        
        # Ordenar para detectar saltos
        selected_indices.sort()
        
        # Detectar y llenar saltos grandes
        gaps = self._detect_large_gaps(selected_indices)
        
        emergency_frames = []
        for start_gap, end_gap in gaps:
            emergency_additions = self._emergency_exploration(
                frames=[],  # Se pasará desde el contexto
                start_idx=start_gap,
                end_idx=end_gap
            )
            emergency_frames.extend(emergency_additions)
        
        # Combinar frames originales y de emergencia
        all_selected = selected_indices + emergency_frames
        all_selected = sorted(list(set(all_selected)))  # Remover duplicados y ordenar
        
        logger.info(f"Frames originales: {len(selected_indices)}, "
                    f"Frames de emergencia: {len(emergency_frames)}, "
                    f"Total final: {len(all_selected)}")
        
        return all_selected

    
def run_bio_inspired_frame_selection(input_dir: str, output_dir: str,
                                     n_scouts: int = 20, n_workers: int = 40) -> Dict:
    """Función principal para ejecutar selección bio-inspirada"""

    start_time = time.time()

    # Cargar frames desde input_dir
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_files = list(Path(input_dir).glob('*.*'))
    image_files = [p for p in all_files if p.suffix.lower() in valid_exts]
    image_files.sort(key=lambda x: x.name)
    logger.info(f"Cargando {len(image_files)} imágenes desde '{input_dir}'")

    frames: List[np.ndarray] = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"No se pudo leer la imagen: {img_path}")
            continue
        frames.append(img)
    if not frames:
        logger.error("No se cargó ninguna imagen válida. Terminando ejecución.")
        return {}

    # Ejecutar enjambre bio-inspirado
    hive = HiveMind(n_scouts=n_scouts, n_workers=n_workers)
    selected_indices = hive.foraging_cycle(frames)
    logger.info(f"Índices seleccionados: {selected_indices}")

    # Preparar carpeta de salida y subcarpetas
    out_base = Path(output_dir)
    images_dir = out_base / 'images'
    data_dir = out_base / 'data'
    out_base.mkdir(parents=True, exist_ok=True)

    # Limpiar y recrear subcarpetas
    for d in (images_dir, data_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Limpia carpetas '{images_dir}' y '{data_dir}'")

    # Copiar las imágenes seleccionadas a output/images
    selected_names: List[str] = []
    for idx in selected_indices:
        try:
            src = image_files[idx]
            dst = images_dir / src.name
            shutil.copy(src, dst)
            selected_names.append(src.name)
        except IndexError:
            logger.warning(f"Índice fuera de rango al copiar: {idx}")

    # Guardar resumen en JSON dentro de output/data
    elapsed = time.time() - start_time
    summary = {
        'selected_indices': selected_indices,
        'selected_frames': selected_names,
        'n_scouts': n_scouts,
        'n_workers': n_workers,
        'execution_time_s': elapsed
    }
    summary_path = data_dir / 'selection_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Resumen de selección guardado en '{summary_path}'")

    return summary

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    # Importa la función principal (asume que está en el mismo archivo o en un módulo llamado `bio_frame_selector`)
    # from bio_frame_selector import run_bio_inspired_frame_selection

    parser = argparse.ArgumentParser(
        description="Ejecuta la selección bio-inspirada de frames para SFM"
    )
    parser.add_argument(
        "un_numero",
        type=str,
        help="Identificador de la carpeta base (p.ej. '001', '002', etc.)"
    )
    parser.add_argument(
        "--n_scouts",
        type=int,
        default=20,
        help="Número de abejas exploradoras (por defecto: 20)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=40,
        help="Número de abejas trabajadoras (por defecto: 40)"
    )
    args = parser.parse_args()

    # Construye rutas basadas en la variable UN_NUMERO
    base = args.un_numero
    input_dir = Path(f"{base}/images/data")
    output_dir = Path(f"{base}/output")

    # Ejecuta la selección de frames
    summary = run_bio_inspired_frame_selection(
        str(input_dir),
        str(output_dir),
        n_scouts=args.n_scouts,
        n_workers=args.n_workers
    )

    # Muestra el resumen en pantalla
    print(json.dumps(summary, indent=2, ensure_ascii=False))