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
    """Get advanced components."""
    return {}

__all__ = [
    'envs',
    'algos',
    'models',
    'registry',
    'utils',
    'get_advanced_components'
]