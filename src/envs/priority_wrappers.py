"""
Environment wrapper that adds priority-based features to existing environments.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple

from src.envs.base_env import Point
from src.envs.colinear import NoThreeCollinearEnv
from src.priority import priority_grid, point_collinear_count


class PriorityEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds priority information to environment observations and info.
    """
    
    def __init__(self, env: NoThreeCollinearEnv, include_priority_in_obs: bool = False):
        """
        Initialize priority wrapper.
        
        Args:
            env: Base environment to wrap
            include_priority_in_obs: Whether to include priority grid in observation
        """
        super().__init__(env)
        self.include_priority_in_obs = include_priority_in_obs
        self.grid_size = max(env.grid_shape)
        
        # Precompute priority grid for efficiency
        self._priority_grid = priority_grid(self.grid_size)
        
        # Update observation space if including priority
        if include_priority_in_obs:
            m, n = env.grid_shape
            # Stack original observation with priority grid
            self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(2, m, n), dtype=np.float32
            )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and add priority information."""
        obs, info = self.env.reset(**kwargs)
        
        # Add priority information to info
        info["priority_grid"] = self._priority_grid.copy()
        info["available_priorities"] = self._get_available_priorities()
        
        # Modify observation if needed
        if self.include_priority_in_obs:
            # Normalize priority grid to [0, 1] range
            norm_priority = self._normalize_priority_grid()
            obs = np.stack([obs, norm_priority], axis=0)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and add priority information."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Add priority-based information to info
        info["priority_grid"] = self._priority_grid.copy()
        info["available_priorities"] = self._get_available_priorities()
        
        if not done:
            point = self.env.decode_action(action)
            info["selected_point_priority"] = self._get_point_priority(point)
        
        # Modify observation if needed
        if self.include_priority_in_obs:
            norm_priority = self._normalize_priority_grid()
            obs = np.stack([obs, norm_priority], axis=0)
        
        return obs, reward, done, truncated, info
    
    def _get_available_priorities(self) -> np.ndarray:
        """Get priorities for all available (unselected) points."""
        priorities = []
        m, n = self.env.grid_shape
        
        for x in range(n):
            for y in range(m):
                point = Point(x, y)
                if not self.env.is_selected(point):
                    priorities.append(self._priority_grid[x, y])
        
        return np.array(priorities)
    
    def _get_point_priority(self, point: Point) -> float:
        """Get priority for a specific point."""
        return self._priority_grid[point.x, point.y]
    
    def _normalize_priority_grid(self) -> np.ndarray:
        """Normalize priority grid to [0, 1] range."""
        p_grid = self._priority_grid
        p_min, p_max = p_grid.min(), p_grid.max()
        if p_max > p_min:
            return (p_grid - p_min) / (p_max - p_min)
        else:
            return np.zeros_like(p_grid)


class PriorityRewardWrapper(gym.Wrapper):
    """
    Wrapper that modifies rewards based on priority values.
    """
    
    def __init__(self, env: NoThreeCollinearEnv, priority_weight: float = 0.1):
        """
        Initialize priority reward wrapper.
        
        Args:
            env: Base environment to wrap
            priority_weight: Weight for priority component in reward
        """
        super().__init__(env)
        self.priority_weight = priority_weight
        self.grid_size = max(env.grid_shape)
        self._priority_grid = priority_grid(self.grid_size)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and modify reward based on priority."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        if not done:
            # Add priority-based reward component
            point = self.env.decode_action(action)
            priority_reward = self._priority_grid[point.x, point.y] * self.priority_weight
            reward += priority_reward
            
            info["base_reward"] = reward - priority_reward
            info["priority_reward"] = priority_reward
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        return self.env.reset(**kwargs)
