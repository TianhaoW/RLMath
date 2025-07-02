"""
MCTS Trainer that integrates with the new environment structure.
Uses GridSubsetEnvWithPriority for consistency with the rest of the codebase.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from src.envs.base_env import Point, GridSubsetEnvWithPriority
from src.geometry import are_collinear
from ..geometry import are_collinear


@dataclass
class MCTSConfig:
    """Configuration for MCTS algorithm."""
    num_searches: int = 1000
    c_param: float = 1.4
    use_priority: bool = True
    top_n_priority: int = 1
    max_rollout_depth: int = 100
    virtual_loss: float = 1.0
    num_workers: int = 4
    use_parallel: bool = False
    show_progress: bool = True


class MCTSNode:
    """MCTS Node for tree search."""
    
    def __init__(self, env: GridSubsetEnvWithPriority, state: np.ndarray, 
                 parent: Optional['MCTSNode'] = None, action: Optional[int] = None):
        self.env = env
        self.state = state.copy()
        self.parent = parent
        self.action = action
        
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value_sum = 0.0
        
        # Reconstruct points from state
        self.points = self._get_points_from_state()
        self.untried_actions = self._get_valid_actions()
        
        # For virtual loss in parallel MCTS
        self.virtual_losses = 0
        
    def _get_points_from_state(self) -> List[Point]:
        """Extract points from the current state."""
        points = []
        m, n = self.env.grid_shape
        for x in range(n):
            for y in range(m):
                if self.state[y, x] == 1:
                    points.append(Point(x, y))
        return points
        
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid actions from current state."""
        valid_actions = []
        m, n = self.env.grid_shape
        
        for action in range(m * n):
            point = self.env.decode_action(action)
            if not self.env.is_selected(point):
                # Check if this action would create three collinear points
                if self._is_valid_action(point):
                    valid_actions.append(action)
        
        # If using priority, filter to top priority actions
        if self.env.priority_fn and hasattr(self.env, 'priority_map'):
            valid_actions = self._filter_by_priority(valid_actions)
            
        return valid_actions
    
    def _is_valid_action(self, point: Point) -> bool:
        """Check if placing a point would create three collinear points."""
        # Check against all existing points
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points[i+1:], start=i+1):
                if are_collinear(p1, p2, point):
                    return False
        return True
    
    def _filter_by_priority(self, actions: List[int]) -> List[int]:
        """Filter actions to keep only those with highest priority."""
        if not actions:
            return actions
            
        # Get priorities for all valid actions
        priorities = []
        for action in actions:
            point = self.env.decode_action(action)
            priority = self.env.priority_map[point.y, point.x]
            if np.isfinite(priority):  # Only consider finite priorities
                priorities.append((action, priority))
        
        if not priorities:
            return []
            
        # Sort by priority (highest first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only top priority actions
        max_priority = priorities[0][1]
        top_actions = [action for action, priority in priorities if priority == max_priority]
        
        return top_actions
    
    def is_fully_expanded(self) -> bool:
        """Check if all valid actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return len(self._get_valid_actions()) == 0
    
    def select_child(self) -> 'MCTSNode':
        """Select best child using UCB1."""
        if not self.children:
            return None
            
        best_child = None
        best_ucb = -np.inf
        
        log_n = np.log(self.visits)
        
        for child in self.children:
            if child.visits == 0:
                return child  # Always select unvisited children first
                
            # UCB1 formula
            exploitation = child.value_sum / child.visits
            exploration = np.sqrt(log_n / child.visits)
            ucb = exploitation + MCTSConfig.c_param * exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
                
        return best_child
    
    def expand(self) -> Optional['MCTSNode']:
        """Expand by adding a new child."""
        if not self.untried_actions:
            return None
            
        # Randomly select an untried action
        action = np.random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        
        # Create new state by applying action
        new_state = self.state.copy()
        point = self.env.decode_action(action)
        new_state[point.y, point.x] = 1
        
        # Create child node
        child = MCTSNode(self.env, new_state, parent=self, action=action)
        self.children.append(child)
        
        return child
    
    def simulate(self) -> float:
        """Simulate random rollout from current state."""
        simulation_env = GridSubsetEnvWithPriority(
            *self.env.grid_shape, 
            priority_fn=self.env.priority_fn
        )
        
        # Set up simulation state
        simulation_env.state = self.state.copy()
        simulation_env.priority_map = self.env.priority_map.copy()
        simulation_env.points = []
        
        # Reconstruct points list from state
        m, n = self.env.grid_shape
        for x in range(n):
            for y in range(m):
                if self.state[y, x] == 1:
                    simulation_env.points.append(Point(x, y))
                    simulation_env.priority_map[y, x] = -np.inf
        
        # Random rollout
        steps = 0
        max_steps = MCTSConfig.max_rollout_depth
        
        while steps < max_steps:
            valid_actions = []
            for action in range(m * n):
                point = simulation_env.decode_action(action)
                if not simulation_env.is_selected(point):
                    # Check validity
                    valid = True
                    for i, p1 in enumerate(simulation_env.points):
                        for j, p2 in enumerate(simulation_env.points[i+1:], start=i+1):
                            if are_collinear(p1, p2, point):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        valid_actions.append(action)
            
            if not valid_actions:
                break
                
            # Apply priority filtering if enabled
            if simulation_env.priority_fn:
                priority_actions = []
                max_priority = -np.inf
                
                for action in valid_actions:
                    point = simulation_env.decode_action(action)
                    priority = simulation_env.priority_map[point.y, point.x]
                    if np.isfinite(priority):
                        if priority > max_priority:
                            max_priority = priority
                            priority_actions = [action]
                        elif priority == max_priority:
                            priority_actions.append(action)
                
                if priority_actions:
                    valid_actions = priority_actions
            
            # Random selection from valid actions
            action = np.random.choice(valid_actions)
            point = simulation_env.decode_action(action)
            
            simulation_env.points.append(point)
            simulation_env.state[point.y, point.x] = 1
            simulation_env.priority_map[point.y, point.x] = -np.inf
            steps += 1
        
        # Return normalized score based on number of points placed
        total_points = len(simulation_env.points)
        max_possible = m * n
        return total_points / max_possible
    
    def backpropagate(self, value: float):
        """Backpropagate value up the tree."""
        self.visits += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backpropagate(value)


class MCTSTrainer:
    """MCTS trainer that integrates with the environment structure."""
    
    def __init__(self, config: Dict[str, Any], env: GridSubsetEnvWithPriority, 
                 model=None, device=None, logger=None):
        self.config = config
        self.env = env
        self.model = model  # Not used for MCTS but kept for interface compatibility
        self.device = device
        self.logger = logger
        
        # MCTS specific configuration
        self.mcts_config = MCTSConfig(
            num_searches=config.get('train', {}).get('mcts_searches', 1000),
            c_param=config.get('train', {}).get('mcts_c_param', 1.4),
            use_priority=config.get('train', {}).get('mcts_use_priority', True),
            top_n_priority=config.get('train', {}).get('mcts_top_n', 1),
            show_progress=config.get('train', {}).get('mcts_progress', True)
        )
        
        self.best_score = 0
        self.best_points = []
        
    def mcts_search(self, root_state: np.ndarray) -> Tuple[int, float]:
        """Perform MCTS search and return best action and its value."""
        root = MCTSNode(self.env, root_state)
        
        if self.logger:
            self.logger.info(f"Starting MCTS search with {self.mcts_config.num_searches} iterations")
        
        # MCTS iterations
        for i in range(self.mcts_config.num_searches):
            node = root
            
            # Selection phase
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child()
                if node is None:
                    break
            
            # Expansion phase
            if node and not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation phase
            if node:
                value = node.simulate()
                # Backpropagation phase
                node.backpropagate(value)
        
        # Select best action based on visit counts
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.action, best_child.value_sum / best_child.visits
        else:
            return None, 0.0
    
    def play_game(self) -> Tuple[List[Point], int]:
        """Play a complete game using MCTS."""
        obs, _ = self.env.reset()
        game_points = []
        
        if self.logger:
            self.logger.info("Starting MCTS game")
        
        step = 0
        while not self.env.terminated:
            # Get current state (first channel of observation)
            state = obs[0] if len(obs.shape) == 3 else obs
            
            # Perform MCTS search
            action, value = self.mcts_search(state)
            
            if action is None:
                if self.logger:
                    self.logger.info("No valid actions available, game ended")
                break
            
            # Take action
            point = self.env.decode_action(action)
            obs, reward, done, truncated, info = self.env.step(action)
            game_points.append(point)
            
            step += 1
            if self.logger:
                self.logger.info(f"Step {step}: Placed point {point}, reward: {reward:.3f}")
            
            if done or truncated:
                break
        
        score = len(game_points)
        if self.logger:
            self.logger.info(f"Game completed with {score} points")
        
        return game_points, score
    
    def train(self):
        """Train using MCTS (play multiple games)."""
        num_episodes = self.config.get('train', {}).get('episodes', 100)
        
        if self.logger:
            self.logger.info(f"Starting MCTS training for {num_episodes} episodes")
        
        scores = []
        all_games = []
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Play one game
            points, score = self.play_game()
            scores.append(score)
            all_games.append(points)
            
            # Track best performance
            if score > self.best_score:
                self.best_score = score
                self.best_points = points.copy()
                if self.logger:
                    self.logger.info(f"New best score: {score} points")
            
            duration = time.time() - start_time
            avg_score = np.mean(scores[-100:])  # Average of last 100 games
            
            if self.logger and (episode + 1) % 10 == 0:
                self.logger.info(
                    f"Episode {episode + 1}/{num_episodes}: "
                    f"Score: {score}, "
                    f"Best: {self.best_score}, "
                    f"Avg (last 100): {avg_score:.2f}, "
                    f"Time: {duration:.2f}s"
                )
        
        # Final results
        if self.logger:
            self.logger.info("MCTS training completed")
            self.logger.info(f"Best score achieved: {self.best_score}")
            self.logger.info(f"Average score: {np.mean(scores):.2f}")
            self.logger.info(f"Best configuration: {self.best_points}")
            
            # Optionally save best points if configured
            if self.config.get('train', {}).get('save_best_points', False):
                self.logger.info("Best point configuration:")
                for i, point in enumerate(self.best_points):
                    self.logger.info(f"  Point {i+1}: {point}")
        
        return {
            'scores': scores,
            'best_score': self.best_score,
            'best_points': self.best_points,
            'all_games': all_games
        }
    
    def evaluate(self, num_games: int = 10) -> Dict[str, Any]:
        """Evaluate MCTS performance over multiple games."""
        if self.logger:
            self.logger.info(f"Evaluating MCTS over {num_games} games")
        
        scores = []
        for game in range(num_games):
            points, score = self.play_game()
            scores.append(score)
            
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        results = {
            'average_score': avg_score,
            'std_score': std_score,
            'max_score': max_score,
            'min_score': min_score,
            'scores': scores
        }
        
        if self.logger:
            self.logger.info(f"Evaluation results:")
            self.logger.info(f"  Average score: {avg_score:.2f} ± {std_score:.2f}")
            self.logger.info(f"  Range: {min_score} - {max_score}")
        
        return results
