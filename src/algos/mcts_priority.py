"""
Mimport numpy as np
import random
from typing import List, Callable, Optional, Tuple
from collections import defaultdict

from src.envs.base_env import Point
from src.geometry import are_collinear
from src.priority import collinear_count_priorityh priority-based selection for no-three-collinear problems.
Extracted from luoning's MCTS along priority notebook.
"""

import numpy as np
import random
from typing import List, Callable, Optional, Tuple
from collections import defaultdict

from ..geometry import Point, are_collinear
from ..priority import collinear_count_priority


class MCTSPriorityNode:
    """Node in MCTS tree with priority-based selection."""
    
    def __init__(self, point: Optional[Point] = None, parent: Optional['MCTSPriorityNode'] = None):
        self.point = point
        self.parent = parent
        self.children: List['MCTSPriorityNode'] = []
        self.visits = 0
        self.wins = 0
        self.untried_moves: List[Point] = []
        
    def add_child(self, point: Point) -> 'MCTSPriorityNode':
        """Add a child node for the given point."""
        child = MCTSPriorityNode(point=point, parent=self)
        self.children.append(child)
        return child
        
    def update(self, result: float):
        """Update node statistics with simulation result."""
        self.visits += 1
        self.wins += result
        
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
        
    def best_child(self, c_param: float = 1.4) -> 'MCTSPriorityNode':
        """Select best child using UCB1 formula."""
        choices_weights = [
            (child.wins / child.visits) + c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]


class MCTSPriorityAgent:
    """MCTS agent with priority-based move selection."""
    
    def __init__(self, grid_size: int, max_iterations: int = 1000, c_param: float = 1.4):
        self.grid_size = grid_size
        self.max_iterations = max_iterations
        self.c_param = c_param
        self.priority_fn = collinear_count_priority(grid_size)
        
    def get_valid_moves(self, current_points: List[Point]) -> List[Point]:
        """Get all valid moves (points that don't create three collinear points)."""
        valid_moves = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                point = Point(x, y)
                if point not in current_points:
                    # Check if adding this point creates any collinear triple
                    valid = True
                    for i in range(len(current_points)):
                        for j in range(i + 1, len(current_points)):
                            if are_collinear(current_points[i], current_points[j], point):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        valid_moves.append(point)
        return valid_moves
    
    def select_move_by_priority(self, valid_moves: List[Point]) -> Point:
        """Select move based on priority function."""
        if not valid_moves:
            return None
            
        # Calculate priorities for all valid moves
        priorities = [self.priority_fn(move) for move in valid_moves]
        
        # Use softmax to convert priorities to probabilities
        priorities = np.array(priorities)
        exp_priorities = np.exp(priorities - np.max(priorities))  # Numerical stability
        probabilities = exp_priorities / np.sum(exp_priorities)
        
        # Sample based on probabilities
        selected_idx = np.random.choice(len(valid_moves), p=probabilities)
        return valid_moves[selected_idx]
    
    def simulate_random_game(self, current_points: List[Point]) -> float:
        """Simulate a random game from current state."""
        points = current_points.copy()
        
        while True:
            valid_moves = self.get_valid_moves(points)
            if not valid_moves:
                break
                
            # Use priority-based selection in simulation
            next_move = self.select_move_by_priority(valid_moves)
            if next_move is None:
                break
                
            points.append(next_move)
            
            # Early termination if we've placed many points
            if len(points) >= self.grid_size * self.grid_size * 0.8:
                break
                
        return len(points)  # Return number of points placed as score
    
    def mcts_search(self, root_points: List[Point]) -> Point:
        """Perform MCTS search to find best move."""
        root = MCTSPriorityNode()
        root.untried_moves = self.get_valid_moves(root_points)
        
        for _ in range(self.max_iterations):
            node = root
            path = []
            current_points = root_points.copy()
            
            # Selection phase
            while not node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                path.append(node)
                current_points.append(node.point)
            
            # Expansion phase
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                node = node.add_child(move)
                path.append(node)
                current_points.append(move)
            
            # Simulation phase
            score = self.simulate_random_game(current_points)
            
            # Backpropagation phase
            for node in path:
                node.update(score)
        
        # Return best move
        if root.children:
            best_child = root.best_child(c_param=0)  # Exploit only
            return best_child.point
        else:
            return self.select_move_by_priority(root.untried_moves)
    
    def play_game(self, verbose: bool = False) -> Tuple[List[Point], int]:
        """Play a complete game using MCTS with priority."""
        points = []
        
        while True:
            valid_moves = self.get_valid_moves(points)
            if not valid_moves:
                break
                
            next_move = self.mcts_search(points)
            if next_move is None:
                break
                
            points.append(next_move)
            
            if verbose:
                print(f"Move {len(points)}: {next_move}")
                
        return points, len(points)
