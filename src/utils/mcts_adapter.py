#!/usr/bin/env python3
"""
MCTS Environment Adapter

This adapter converts the RLMath environment interface to the interface
expected by the MCTS implementations extracted from notebooks.
"""

import numpy as np
from numba import njit


class MCTSEnvironmentAdapter:
    """
    Adapter that provides the interface expected by MCTS implementations
    while using the actual RLMath environments underneath.
    """
    
    def __init__(self, env):
        self.env = env
        self.m, self.n = env.grid_shape
        self.row_count = self.m
        self.column_count = self.n
        self.action_size = self.m * self.n
        self.pts_upper_bound = min(self.m, self.n) * 2  # Reasonable upper bound
        
        # Priority grid if needed
        self.priority_grid = None
    
    def get_initial_state(self):
        """Get initial state as numpy array."""
        obs, _ = self.env.reset()
        return obs.astype(np.uint8)
    
    def get_next_state(self, state, action):
        """Apply action to state and return new state."""
        new_state = state.copy()
        row = action // self.column_count
        col = action % self.column_count
        new_state[row, col] = 1
        return new_state
    
    def get_valid_moves(self, state):
        """Get valid moves as a binary mask."""
        # Create temporary environment state
        temp_env = type(self.env)(self.m, self.n)
        temp_env.state = state.astype(np.float32)
        
        # Reconstruct points list from state
        temp_env.points = []
        for i in range(self.m):
            for j in range(self.n):
                if state[i, j] == 1:
                    from src.envs.base_env import Point
                    temp_env.points.append(Point(j, i))
        
        # For NoThreeCollinearEnv, reconstruct slope_map
        if hasattr(temp_env, 'slope_map'):
            from collections import defaultdict
            from src.envs.colinear import reduced_slope
            temp_env.slope_map = defaultdict(set)
            
            # Rebuild slope map
            for i, p1 in enumerate(temp_env.points):
                for j, p2 in enumerate(temp_env.points):
                    if i != j:
                        slope = reduced_slope(p1, p2)
                        temp_env.slope_map[p1].add(slope)
        
        # Check each empty position
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        
        for action in range(self.action_size):
            row = action // self.column_count
            col = action % self.column_count
            
            if state[row, col] == 0:  # Empty position
                from src.envs.base_env import Point
                point = Point(col, row)
                
                # Check if adding this point would be valid
                is_valid = True
                if hasattr(temp_env, 'slope_map'):
                    # Check collinearity for NoThreeCollinearEnv
                    from src.envs.colinear import reduced_slope
                    for p in temp_env.points:
                        slope = reduced_slope(p, point)
                        if slope in temp_env.slope_map[p]:
                            is_valid = False
                            break
                
                if is_valid:
                    valid_moves[action] = 1
        
        return valid_moves
    
    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken):
        """Get valid moves for child state (incremental update)."""
        # For now, use the full calculation - could be optimized
        new_state = self.get_next_state(parent_state, action_taken)
        return self.get_valid_moves(new_state)
    
    def get_value_and_terminated(self, state, valid_moves=None):
        """Get value and termination status."""
        if valid_moves is None:
            valid_moves = self.get_valid_moves(state)
            
        total_valid = np.sum(valid_moves)
        
        if total_valid == 0:
            # Game ended, calculate final value
            num_points = np.sum(state)
            value = num_points / self.pts_upper_bound  # Normalized value
            return value, True
        else:
            # Game continues
            return 0.0, False
    
    def check_collinear(self, state, action=None):
        """Check for collinear triples (mainly for compatibility)."""
        # This is mainly used for debugging/validation
        if action is not None:
            temp_state = state.copy()
            row = action // self.column_count
            col = action % self.column_count
            temp_state[row, col] = 1
        else:
            temp_state = state
        
        # Count collinear triples - simplified implementation
        points = []
        for i in range(self.m):
            for j in range(self.n):
                if temp_state[i, j] == 1:
                    points.append((j, i))  # (x, y) format
        
        collinear_count = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    p1, p2, p3 = points[i], points[j], points[k]
                    if self._are_collinear(p1, p2, p3):
                        collinear_count += 1
        
        return collinear_count
    
    def _are_collinear(self, p1, p2, p3):
        """Check if three points are collinear."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)
    
    def display_state(self, state, action_prob=None):
        """Display state (delegated to environment)."""
        # Create temporary environment for display
        temp_env = type(self.env)(self.m, self.n)
        temp_env.state = state.astype(np.float32)
        
        # Reconstruct points
        temp_env.points = []
        for i in range(self.m):
            for j in range(self.n):
                if state[i, j] == 1:
                    from src.envs.base_env import Point
                    temp_env.points.append(Point(j, i))
        
        temp_env.plot()


# Numba-compiled functions for compatibility
@njit(cache=True)
def _are_collinear_nb(x1, y1, x2, y2, x3, y3):
    """Numba version of collinearity check."""
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)


@njit(cache=True) 
def get_valid_moves_nb(state, row_count, column_count):
    """Numba-optimized valid moves calculation."""
    max_pts = row_count * column_count
    coords = np.empty((max_pts, 2), np.int64)
    n_pts = 0

    # Collect coordinates of existing points
    for i in range(row_count):
        for j in range(column_count):
            if state[i, j] == 1:
                coords[n_pts, 0] = i
                coords[n_pts, 1] = j
                n_pts += 1

    mask = np.zeros(row_count * column_count, np.uint8)

    # Check each empty cell
    for i in range(row_count):
        for j in range(column_count):
            if state[i, j] != 0:
                continue
            valid = True
            # Check for collinearity with every pair of existing points
            for p in range(n_pts):
                for q in range(p + 1, n_pts):
                    i1, j1 = coords[p, 0], coords[p, 1]
                    i2, j2 = coords[q, 0], coords[q, 1]
                    if _are_collinear_nb(j1, i1, j2, i2, j, i):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                mask[i * column_count + j] = 1
    return mask


@njit(cache=True)
def get_valid_moves_subset_nb(parent_state, parent_valid_moves, action_taken, row_count, column_count):
    """Numba-optimized incremental valid moves update."""
    # Copy input mask and remove the taken action
    mask = parent_valid_moves.copy()
    mask[action_taken] = 0

    # Coordinates of the newly placed point
    new_r = action_taken // column_count
    new_c = action_taken % column_count

    # Iterate over all existing points
    for pr in range(row_count):
        for pc in range(column_count):
            if not parent_state[pr, pc]:
                continue
            # Skip the new point itself
            if pr == new_r and pc == new_c:
                continue

            dr = pr - new_r
            dc = pc - new_c

            # Infinite slope (vertical line): invalidate entire column
            if dc == 0:
                for rr in range(row_count):
                    idx = rr * column_count + new_c
                    mask[idx] = 0
                continue

            # Zero slope (horizontal line): invalidate entire row
            if dr == 0:
                row_index = pr
                base = row_index * column_count
                for cc in range(column_count):
                    mask[base + cc] = 0
                continue

            # General case: remove points on the infinite line
            for cc in range(column_count):
                num = (cc - new_c) * dr
                if num % dc != 0:
                    continue
                rr = new_r + num // dc
                if rr < 0 or rr >= row_count:
                    continue
                idx = rr * column_count + cc
                mask[idx] = 0

    return mask


@njit(cache=True)
def simulate_nb(state, row_count, column_count, pts_upper_bound):
    """Numba-optimized simulation."""
    max_size = row_count * column_count

    # Get initial valid moves
    valid_moves = get_valid_moves_nb(state, row_count, column_count)
    total_valid = np.sum(valid_moves)

    # Rollout until no moves remain
    while total_valid > 0:
        # Collect valid actions
        acts = np.empty(total_valid, np.int64)
        k = 0
        for idx in range(max_size):
            if valid_moves[idx]:
                acts[k] = idx
                k += 1

        # Pick random action
        pick = acts[np.random.randint(0, total_valid)]

        # Update state and valid moves
        valid_moves = get_valid_moves_subset_nb(
            state, valid_moves, pick, row_count, column_count
        )
        state[pick // column_count, pick % column_count] = 1
        total_valid = np.sum(valid_moves)

    # Return normalized value
    total_points = np.sum(state)
    return total_points / pts_upper_bound


@njit(cache=True)
def get_value_nb(state, pts_upper_bound):
    """Numba-optimized value calculation."""
    total = np.sum(state)
    return total / pts_upper_bound


def create_mcts_adapter(env):
    """Create an MCTS adapter for the given environment."""
    return MCTSEnvironmentAdapter(env)
