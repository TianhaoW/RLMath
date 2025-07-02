"""
No-3-In-Line Environment Implementation
Extracted from nothreeinline-Spring-25 notebooks
"""

import numpy as np
from itertools import combinations
from .base_env import BaseEnv


class NoThreeInLineEnv(BaseEnv):
    """
    No-3-In-Line Environment
    
    The goal is to place as many points as possible on a grid such that
    no three points are collinear.
    """
    
    def __init__(self, grid_size=(3, 10)):
        """
        Initialize the No-3-In-Line environment
        
        Args:
            grid_size: Tuple of (rows, columns) for the grid
        """
        super().__init__()
        self.row_count = grid_size[0]
        self.column_count = grid_size[1]
        self.action_size = self.row_count * self.column_count
        self.grid_size = grid_size
        
    def get_initial_state(self):
        """Return initial empty state"""
        return np.zeros((self.row_count, self.column_count), dtype=np.int8)
    
    def get_next_state(self, state, action):
        """
        Get next state after placing a point
        
        Args:
            state: Current grid state
            action: Action (position to place point)
            
        Returns:
            Updated state
        """
        row = action // self.column_count
        column = action % self.column_count
        new_state = state.copy()
        new_state[row, column] = 1
        return new_state
    
    def get_valid_moves(self, state):
        """
        Get valid moves (empty positions)
        
        Args:
            state: Current grid state
            
        Returns:
            Binary array indicating valid moves
        """
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_collinear(self, state, action):
        """
        Check if placing a point at action would create collinear triples
        
        Args:
            state: Current grid state
            action: Action to check
            
        Returns:
            Number of collinear triples that would be created
        """
        row = action // self.column_count
        column = action % self.column_count
        state_next = state.copy()
        state_next[row, column] = 1

        # Get coordinates of all points with value 1
        coords = np.argwhere(state_next == 1)
        
        # Get all combinations of 3 points
        triples = list(combinations(coords, 3))

        number_of_collinear_triples = 0
        for triple in triples:
            if self._are_collinear(triple[0], triple[1], triple[2]):
                number_of_collinear_triples += 1
        
        return number_of_collinear_triples
    
    def _are_collinear(self, p1, p2, p3):
        """
        Check if three points are collinear
        Adapted from https://github.com/kitft/funsearch
        
        Args:
            p1, p2, p3: Points as (x, y) tuples or arrays
            
        Returns:
            True if points are collinear
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)
    
    def get_value_and_terminated(self, state):
        """
        Get the value (number of points) and termination status
        
        Args:
            state: Current grid state
            
        Returns:
            Tuple of (value, terminated)
        """
        value = np.sum(state.reshape(-1) == 1)
        
        # Check if any valid move would create collinear triples
        valid_moves = self.get_valid_moves(state)
        valid_actions = np.where(valid_moves == 1)[0]
        
        terminated = True
        for action in valid_actions:
            if self.check_collinear(state, action) == 0:
                terminated = False
                break
                
        return value, terminated
    
    def is_terminal(self, state):
        """Check if state is terminal (no valid moves without creating collinear triples)"""
        _, terminated = self.get_value_and_terminated(state)
        return terminated
    
    def get_reward(self, state, action, next_state):
        """
        Get reward for taking action
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Reward value
        """
        # Check if action creates collinear triples
        collinear_count = self.check_collinear(state, action)
        
        if collinear_count > 0:
            # Penalty for creating collinear triples
            return -10 * collinear_count
        else:
            # Reward for placing a valid point
            return 1
    
    def action_to_coordinates(self, action):
        """Convert action to (row, column) coordinates"""
        row = action // self.column_count
        column = action % self.column_count
        return (row, column)
    
    def coordinates_to_action(self, row, column):
        """Convert (row, column) coordinates to action"""
        return row * self.column_count + column
    
    def get_state_points(self, state):
        """Get list of points (coordinates) in the current state"""
        coords = np.argwhere(state == 1)
        return [(int(x), int(y)) for x, y in coords]
    
    def render(self, state):
        """Render the current state"""
        print("Current grid state:")
        print(state)
        points = self.get_state_points(state)
        print(f"Points placed: {len(points)}")
        print(f"Coordinates: {points}")


# Alias for backward compatibility and consistency with existing notebooks
N3il = NoThreeInLineEnv
