"""
Priority functions for grid-based environments.
These functions assign priority scores to grid points to guide search algorithms.
"""

import numpy as np
from src.envs.base_env import Point


def default_priority(point: Point, grid_shape) -> float:
    """Default priority function - no preference."""
    return 0.0


def boundary_priority(point: Point, grid_shape) -> float:
    """Higher priority for points closer to boundaries."""
    m, n = grid_shape
    x, y = point.x, point.y
    
    # Distance to nearest boundary
    min_dist = min(x, y, n - 1 - x, m - 1 - y)
    # Normalize by maximum possible distance to center
    max_dist = min(n // 2, m // 2)
    
    if max_dist == 0:
        return 1.0
    
    # Higher score for points closer to boundary (lower distance)
    return 1.0 - (min_dist / max_dist)


def distance_priority(point: Point, grid_shape) -> float:
    """Priority based on distance from center."""
    m, n = grid_shape
    center_x, center_y = (n - 1) / 2, (m - 1) / 2
    
    # Euclidean distance from center
    dist = np.sqrt((point.x - center_x)**2 + (point.y - center_y)**2)
    
    # Maximum possible distance
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    if max_dist == 0:
        return 1.0
    
    # Normalize to [0, 1]
    return dist / max_dist


def collinear_count_priority(point: Point, grid_shape) -> float:
    """Priority based on potential for creating collinear constraints."""
    m, n = grid_shape
    x, y = point.x, point.y
    
    # Count potential collinear positions
    # This is a simplified heuristic based on position
    
    # Points near corners have fewer potential collinear positions
    corner_distances = [
        np.sqrt(x**2 + y**2),                    # top-left
        np.sqrt((n-1-x)**2 + y**2),              # top-right  
        np.sqrt(x**2 + (m-1-y)**2),              # bottom-left
        np.sqrt((n-1-x)**2 + (m-1-y)**2)        # bottom-right
    ]
    
    min_corner_dist = min(corner_distances)
    max_corner_dist = np.sqrt((n-1)**2 + (m-1)**2)
    
    if max_corner_dist == 0:
        return 1.0
        
    # Higher priority for points further from corners
    return min_corner_dist / max_corner_dist


def create_priority_function(priority_type: str):
    """Create a priority function based on type string."""
    priority_functions = {
        "default": default_priority,
        "boundary": boundary_priority, 
        "distance": distance_priority,
        "collinear_count": collinear_count_priority,
    }
    
    if priority_type not in priority_functions:
        raise ValueError(f"Unknown priority type: {priority_type}. Available: {list(priority_functions.keys())}")
    
    return priority_functions[priority_type]


def create_custom_priority(m: int, n: int, priority_type: str = "boundary"):
    """Create a specific priority function for a grid size."""
    base_fn = create_priority_function(priority_type)
    grid_shape = (m, n)
    
    def priority_fn(point: Point, grid_shape_param=None):
        return base_fn(point, grid_shape)
    
    return priority_fn
