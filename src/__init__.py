"""
RLMath package - Reinforcement Learning for Mathematical Problems
"""

# Import core modules without automatic submodule loading to avoid circular imports
from . import envs
from . import algos
from . import models
from . import registry
from . import utils

# Advanced components (lazy loaded to avoid circular imports)
def get_advanced_components():
    """Get advanced MCTS and priority-based components."""
    from .evaluation_advanced import evaluate_advanced_mcts, demo_single_evaluation, demo_batch_comparison
    from .priority_advanced import ensure_priority_grid_exists, load_priority_grid
    return {
        'evaluate_advanced_mcts': evaluate_advanced_mcts,
        'demo_single_evaluation': demo_single_evaluation,
        'demo_batch_comparison': demo_batch_comparison,
        'ensure_priority_grid_exists': ensure_priority_grid_exists,
        'load_priority_grid': load_priority_grid
    }

__all__ = [
    'envs',
    'algos',
    'models',
    'registry',
    'utils',
    'get_advanced_components'
]