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
    region_x: int
    region_y: int
    quality_estimate: float = 0.0
    explored: bool = False
    exploitation_count: int = 0
    feature_density: float = 0.0
    motion_saliency: float = 0.0
    correspondence_potential: float = 0.0

@dataclass
class FoodSource:
    """
    FoodSource represents a frame with high quality for SFM.
    Contents:
        location, nectar_amount, distance, persistence, scouts_visited
    """
    location: SpatialLocation
    nectar_amount: float  # SFM quality
    distance: float       # Computational cost
    persistence: int = 0
    scouts_visited: Set[int] = field(default_factory=set)
    baseline_quality: float = 0.0
    temporal_coherence: float = 0.0
    
    def get_profitability(self) -> float:
        """
        Input: None
        Context: Calculates the profitability of the food source as quality/cost (bio-realistic).
        Output: Profitability score (float)
        """
        return self.nectar_amount / max(self.distance, 0.1)
