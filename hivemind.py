import threading
import queue
import numpy as np
from typing import List, Dict, Set, Tuple
from bees import ScoutBee, WorkerBee, WaggleDancer
from eye import CompoundEye
from bee_types import FoodSource, SpatialLocation
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from logconfig import logger

class HiveMind:
    """
    Collective mind that coordinates swarm behavior.
    Contents:
        n_scouts, n_workers, compound_eye, scouts, workers, dancers, known_food_sources,
        explored_locations, dance_floor, hive_knowledge, message_queue, lock
    """
    
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
                       for i in range(min(n_scouts, 10))]
        
        # Estado global
        self.known_food_sources: List[FoodSource] = []
        self.explored_locations: Set[int] = set()
        self.dance_floor: List[Dict] = []
        self.hive_knowledge: Dict[str, float] = {}
        
        # Comunicación entre abejas
        self.message_queue = queue.Queue()
        self.lock = threading.Lock()
        
    def foraging_cycle(self, frames: List[np.ndarray], 
                      max_cycles: int = 12) -> List[int]:

        logger.info(f"Starting foraging with {len(frames)} frames")
        
        for cycle in range(max_cycles):
            logger.info(f"Foraging cycle {cycle + 1}/{max_cycles}")
            
            # Execute foraging phases
            self._scout_exploration_phase(frames)
            self._worker_evaluation_phase(frames)
            self._dance_communication_phase()
            self._dance_following_phase(frames)
            
            if cycle >= 5 and self._check_convergence():
                logger.info(f"Convergence reached at cycle {cycle + 1}")
                break
                
        # Final selection with gap management
        selected_frames = self._select_final_frames_with_frames(frames)
        
        logger.info(f"Foraging completed. Selected {len(selected_frames)} frames")
        return selected_frames
    
    def _detect_large_gaps(self, selected_indices: List[int]) -> List[Tuple[int, int]]:
        """
        Input: selected_indices (List[int])
        Context: Detects large gaps between selected frames.
        Output: List of tuples with gap start and end indices (List[Tuple[int, int]])
        """
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
        """
        Input: frames (List[np.ndarray])
        Context: Selects final frames including emergency exploration for large gaps.
        Output: List of selected frame indices (List[int])
        """
        
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
        """
        Input: frames (List[np.ndarray])
        Context: Distributed exploration phase by scout bees.
        Output: None
        """
        
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
        """
        Input: frames (List[np.ndarray])
        Context: Detailed evaluation phase by worker bees.
        Output: None
        """
        
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
        """
        Input: None
        Context: Communication phase via waggle dance.
        Output: None
        """
        
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
        """
        Input: frames (List[np.ndarray])
        Context: Phase where bees follow information from dances.
        Output: None
        """
        
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
        """
        Input: None
        Context: Checks convergence with more relaxed criteria.
        Output: True if converged, False otherwise (bool)
        """
        
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
        """
        Input: None
        Context: Final selection with improved continuity and more permissive threshold.
        Output: List of selected frame indices (List[int])
        """
        
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

