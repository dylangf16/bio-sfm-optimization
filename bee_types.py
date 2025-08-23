from dataclasses import dataclass, field
from enum import Enum
from typing import Set

class BeeBehaviorState(Enum):
    """
    BeeBehaviorState defines the behavioral states of a bee in the swarm.
    Contents:
        EXPLORING, EVALUATING, DANCING, FOLLOWING, INACTIVE
    """
    EXPLORING = "exploring"
    EVALUATING = "evaluating" 
    DANCING = "dancing"
    FOLLOWING = "following"
    INACTIVE = "inactive"

@dataclass
class SpatialLocation:
    """
    Represents a location in the frame space for exploration.
    Contents:
        frame_index, region_x, region_y, quality_estimate, explored, exploitation_count
    """
    frame_index: int
    region_x: int  # Región espacial dentro del frame
    region_y: int
    quality_estimate: float = 0.0
    explored: bool = False
    exploitation_count: int = 0

@dataclass
class FoodSource:
    """
    FoodSource represents a frame with high quality for SFM.
    Contents:
        location, nectar_amount, distance, persistence, scouts_visited
    """
    location: SpatialLocation
    nectar_amount: float  # Calidad SFM del frame
    distance: float       # Costo computacional
    persistence: int = 0  # Cuántos ciclos ha sido visitada
    scouts_visited: Set[int] = field(default_factory=set)
    
    def get_profitability(self) -> float:
        """
        Input: None
        Context: Calculates the profitability of the food source as quality/cost (bio-realistic).
        Output: Profitability score (float)
        """
        return self.nectar_amount / max(self.distance, 0.1)
