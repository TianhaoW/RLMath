"""
Priority-based agent for no-three-collinear environment.
Integrates with existing environment framework.
"""

import numpy as np
from typing import List, Optional, Callable
from src.envs.base_env import Point
from src.envs.colinear import NoThreeCollinearEnv
from src.priority import collinear_count_priority, point_collinear_count


class PriorityAgent:
    """Agent that selects moves based on collinear priority calculations."""
    
    def __init__(self, env: NoThreeCollinearEnv, noise: float = 0.0, temperature: float = 1.0):
        """
        Initialize priority-based agent.
        
        Args:
            env: The no-three-collinear environment
            noise: Random noise to add to priorities
            temperature: Temperature for softmax selection (higher = more random)
        """
        self.env = env
        self.noise = noise
        self.temperature = temperature
        self.grid_size = max(env.grid_shape)  # Assume square grid for now
        self.priority_fn = collinear_count_priority(self.grid_size, noise)
        
    def get_valid_actions(self) -> List[int]:
        """Get all valid actions (encoded) from current environment state."""
        valid_actions = []
        m, n = self.env.grid_shape
        
        for action in range(m * n):
            point = self.env.decode_action(action)
            if not self.env.is_selected(point):
                # Check if this action would create a collinear triple
                would_violate = False
                for i, p1 in enumerate(self.env.points):
                    for j in range(i + 1, len(self.env.points)):
                        p2 = self.env.points[j]
                        # Use existing environment's slope calculation
                        from src.envs.colinear import reduced_slope
                        if reduced_slope(p1, point) == reduced_slope(p1, p2):
                            would_violate = True
                            break
                    if would_violate:
                        break
                
                if not would_violate:
                    valid_actions.append(action)
                    
        return valid_actions
    
    def select_action_greedy(self) -> Optional[int]:
        """Select action greedily based on priority."""
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return None
            
        # Calculate priorities for valid actions
        priorities = []
        for action in valid_actions:
            point = self.env.decode_action(action)
            priority = self.priority_fn(point)
            priorities.append(priority)
        
        # Select action with highest priority
        best_idx = np.argmax(priorities)
        return valid_actions[best_idx]
    
    def select_action_softmax(self) -> Optional[int]:
        """Select action using softmax over priorities."""
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return None
            
        # Calculate priorities for valid actions
        priorities = []
        for action in valid_actions:
            point = self.env.decode_action(action)
            priority = self.priority_fn(point)
            priorities.append(priority)
        
        # Apply temperature and softmax
        priorities = np.array(priorities) / self.temperature
        exp_priorities = np.exp(priorities - np.max(priorities))  # Numerical stability
        probabilities = exp_priorities / np.sum(exp_priorities)
        
        # Sample action based on probabilities
        selected_idx = np.random.choice(len(valid_actions), p=probabilities)
        return valid_actions[selected_idx]
    
    def play_episode(self, max_steps: int = 100, method: str = "softmax", verbose: bool = False) -> int:
        """
        Play a complete episode using priority-based selection.
        
        Args:
            max_steps: Maximum number of steps
            method: Selection method ("greedy" or "softmax")
            verbose: Whether to print step-by-step information
            
        Returns:
            Number of points successfully placed
        """
        obs, _ = self.env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            if method == "greedy":
                action = self.select_action_greedy()
            else:
                action = self.select_action_softmax()
                
            if action is None:
                # No valid moves available
                break
                
            obs, reward, done, truncated, info = self.env.step(action)
            step += 1
            
            if verbose:
                point = self.env.decode_action(action)
                print(f"Step {step}: Selected {point}, Reward: {reward}")
                
        return len(self.env.points)
    
    def evaluate_agent(self, num_episodes: int = 10, method: str = "softmax") -> dict:
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            method: Selection method ("greedy" or "softmax")
            
        Returns:
            Dictionary with evaluation metrics
        """
        scores = []
        
        for episode in range(num_episodes):
            score = self.play_episode(method=method, verbose=False)
            scores.append(score)
            
        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "scores": scores
        }
