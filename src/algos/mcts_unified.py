"""
Unified MCTS Implementation
Integrates all advanced features from notebooks 08_03, 08_04, and 08_05_01
"""

import numpy as np
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from numba import njit
from tqdm.notebook import trange
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
import matplotlib.pyplot as plt

# =============================================================================
# Numba-accelerated Priority Functions
# =============================================================================

@njit(cache=True, nogil=True)
def _supnorm_priority_array_nb(n):
    """Numba-compiled sup-norm priority calculation."""
    arr = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            m = i if i > j else j
            arr[i, j] = float(m)
    return arr

# =============================================================================
# Priority Systems
# =============================================================================

class PrioritySystem(ABC):
    """Abstract base class for priority calculation systems."""
    
    @abstractmethod
    def compute_priority_grid(self, n: int) -> np.ndarray:
        """Compute priority grid for n x n board."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return name of priority system."""
        pass

class SupNormPriority(PrioritySystem):
    """Sup-norm (max distance) priority system."""
    
    def compute_priority_grid(self, n: int) -> np.ndarray:
        return _supnorm_priority_array_nb(n)
    
    def get_name(self) -> str:
        return "SupNorm"

class CollinearCountPriority(PrioritySystem):
    """Collinear count-based priority system (from existing codebase)."""
    
    def compute_priority_grid(self, n: int) -> np.ndarray:
        # This would use the existing priority calculation
        # For now, return a placeholder
        return np.random.random((n, n))
    
    def get_name(self) -> str:
        return "CollinearCount"

class CustomPriority(PrioritySystem):
    """Custom priority system using user-defined priority function."""
    
    def __init__(self, priority_fn, grid_size):
        from src.envs.base_env import Point
        self.priority_fn = priority_fn
        self.grid_size = grid_size
        self.Point = Point
    
    def compute_priority_grid(self, n: int) -> np.ndarray:
        """Compute priority grid using custom function."""
        arr = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                point = self.Point(j, i)  # Note: Point(x, y) where x=col, y=row
                arr[i, j] = self.priority_fn(point, self.grid_size)
        return arr
    
    def get_name(self) -> str:
        return "Custom"

# =============================================================================
# Numba-accelerated Game Functions
# =============================================================================

@njit(cache=True, nogil=True)
def value_fn_nb(x):
    """Value function for scoring."""
    return x

@njit(cache=True, nogil=True)
def get_value_nb(state, pts_upper_bound):
    """Compute normalized game value."""
    total = np.sum(state)
    n = pts_upper_bound / 2
    return (total - 1.5 * n) / (0.5 * n)

@njit(cache=True, nogil=True)
def _are_collinear(x1, y1, x2, y2, x3, y3):
    """Check if three points are collinear."""
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

@njit(cache=True, nogil=True)
def get_valid_moves_nb(state, row_count, column_count):
    """Get valid moves avoiding collinear placements."""
    max_pts = row_count * column_count
    coords = np.empty((max_pts, 2), np.int64)
    n_pts = 0
    
    # Collect existing points
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
            # Check collinearity with all pairs of existing points
            for p in range(n_pts):
                for q in range(p + 1, n_pts):
                    i1, j1 = coords[p, 0], coords[p, 1]
                    i2, j2 = coords[q, 0], coords[q, 1]
                    if _are_collinear(j1, i1, j2, i2, j, i):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                mask[i * column_count + j] = 1
    return mask

@njit(cache=True, nogil=True)
def get_valid_moves_subset_nb(parent_state, parent_valid_moves, action_taken, row_count, column_count):
    """Incrementally update valid moves after placing a point."""
    mask = parent_valid_moves.copy()
    mask[action_taken] = 0
    
    new_r = action_taken // column_count
    new_c = action_taken % column_count
    
    # Invalidate collinear positions
    for pr in range(row_count):
        for pc in range(column_count):
            if not parent_state[pr, pc]:
                continue
            if pr == new_r and pc == new_c:
                continue
                
            dr = pr - new_r
            dc = pc - new_c
            
            # Vertical line
            if dc == 0:
                for rr in range(row_count):
                    idx = rr * column_count + new_c
                    mask[idx] = 0
                continue
            
            # Horizontal line
            if dr == 0:
                base = pr * column_count
                for cc in range(column_count):
                    mask[base + cc] = 0
                continue
            
            # General case
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

@njit(cache=True, nogil=True)
def filter_top_priority_moves(valid_moves, priority_grid, row_count, column_count, top_N=1):
    """Filter valid moves to top N priority levels."""
    indices = []
    priorities = []
    
    for idx in range(valid_moves.shape[0]):
        if valid_moves[idx] == 1:
            indices.append(idx)
            i = idx // column_count
            j = idx % column_count
            priorities.append(priority_grid[i, j])
    
    if len(indices) == 0:
        return valid_moves
    
    # Manual sort for Numba compatibility
    n = len(priorities)
    unique_priorities = []
    for k in range(n):
        p = priorities[k]
        found = False
        for l in range(len(unique_priorities)):
            if unique_priorities[l] == p:
                found = True
                break
        if not found:
            unique_priorities.append(p)
    
    # Sort descending
    for i in range(len(unique_priorities)):
        max_idx = i
        for j in range(i+1, len(unique_priorities)):
            if unique_priorities[j] > unique_priorities[max_idx]:
                max_idx = j
        tmp = unique_priorities[i]
        unique_priorities[i] = unique_priorities[max_idx]
        unique_priorities[max_idx] = tmp
    
    # Select top N priorities
    N = min(top_N, len(unique_priorities))
    threshold = unique_priorities[:N]
    
    # Build filtered mask
    mask = np.zeros_like(valid_moves)
    for k in range(n):
        idx = indices[k]
        p = priorities[k]
        for t in range(N):
            if p == threshold[t]:
                mask[idx] = 1
                break
    return mask

@njit(cache=True, nogil=True)
def simulate_nb(state, row_count, column_count, pts_upper_bound):
    """Standard random rollout simulation."""
    max_size = row_count * column_count
    valid_moves = get_valid_moves_nb(state, row_count, column_count)
    total_valid = np.sum(valid_moves)
    
    while total_valid > 0:
        acts = np.empty(total_valid, np.int64)
        k = 0
        for idx in range(max_size):
            if valid_moves[idx]:
                acts[k] = idx
                k += 1
        
        pick = acts[np.random.randint(0, total_valid)]
        valid_moves = get_valid_moves_subset_nb(state, valid_moves, pick, row_count, column_count)
        
        r = pick // column_count
        c = pick % column_count
        state[r, c] = 1
        
        total_valid = np.sum(valid_moves)
    
    return get_value_nb(state, pts_upper_bound)

@njit(cache=True, nogil=True)
def simulate_with_priority_nb(state, row_count, column_count, pts_upper_bound, priority_grid, top_N):
    """Priority-guided rollout simulation."""
    max_size = row_count * column_count
    valid_moves = get_valid_moves_nb(state, row_count, column_count)
    valid_moves = filter_top_priority_moves(valid_moves, priority_grid, row_count, column_count, top_N)
    total_valid = np.sum(valid_moves)
    
    while total_valid > 0:
        acts = np.empty(total_valid, np.int64)
        k = 0
        for idx in range(max_size):
            if valid_moves[idx]:
                acts[k] = idx
                k += 1
        
        pick = acts[np.random.randint(0, total_valid)]
        valid_moves = get_valid_moves_subset_nb(state, valid_moves, pick, row_count, column_count)
        state[pick // column_count, pick % column_count] = 1
        
        valid_moves = filter_top_priority_moves(valid_moves, priority_grid, row_count, column_count, top_N)
        total_valid = np.sum(valid_moves)
    
    return get_value_nb(state, pts_upper_bound)

# =============================================================================
# Game Environment
# =============================================================================

class N3ilUnified:
    """Unified No-Three-In-Line environment with all advanced features."""
    
    def __init__(self, grid_size, args, priority_system: Optional[PrioritySystem] = None):
        self.row_count, self.column_count = grid_size
        self.pts_upper_bound = np.min(grid_size) * 2
        self.action_size = self.row_count * self.column_count
        self.args = args
        
        # Set up priority system
        if priority_system is None:
            priority_system = SupNormPriority()
        self.priority_system = priority_system
        self.priority_grid = priority_system.compute_priority_grid(self.row_count)
    
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), np.uint8)
    
    def get_next_state(self, state, action):
        row = action // self.column_count
        col = action % self.column_count
        state[row, col] = 1
        return state
    
    def get_valid_moves(self, state):
        valid_moves = get_valid_moves_nb(state, self.row_count, self.column_count)
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, self.priority_grid, 
                self.row_count, self.column_count, 
                top_N=self.args.get('top_n', 1)
            )
        return valid_moves
    
    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken):
        valid_moves = get_valid_moves_subset_nb(
            parent_state, parent_valid_moves, action_taken, 
            self.row_count, self.column_count
        )
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, self.priority_grid,
                self.row_count, self.column_count,
                top_N=self.args.get('top_n', 1)
            )
        return valid_moves
    
    def get_value_and_terminated(self, state, valid_moves):
        if np.sum(valid_moves) > 0:
            return 0.0, False
        value = get_value_nb(state, self.pts_upper_bound)
        return value, True
    
    def get_encoded_state(self, state):
        encoded_state = np.stack((state == 0, state == 1)).astype(np.float32)
        return encoded_state

# =============================================================================
# Advanced MCTS Node
# =============================================================================

class AdvancedNode:
    """Enhanced MCTS node with all advanced features."""
    
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.env = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)
        
        # Compute valid moves
        if parent is None:
            self.valid_moves = game.get_valid_moves(state)
        else:
            self.valid_moves = game.get_valid_moves_subset(
                parent.state, parent.valid_moves, self.action_taken
            )
        
        self.is_full = False
        self._cached_ucb = None
        self._ucb_dirty = True
    
    def apply_virtual_loss(self):
        """Apply virtual loss for parallel processing."""
        with self.lock:
            self.value_sum -= self._vl
            self.visit_count += 1
            self._ucb_dirty = True
    
    def revert_virtual_loss(self):
        """Revert virtual loss after simulation."""
        with self.lock:
            self.value_sum += self._vl
            self._ucb_dirty = True
    
    def is_fully_expanded(self):
        return self.is_full and len(self.children) > 0
    
    def select(self, iteration=0):
        """Select best child using UCB with optional annealing."""
        best_child = None
        best_ucb = -np.inf
        log_N = math.log(self.visit_count)
        
        for child in self.children:
            ucb = self.get_ucb(child, iteration, log_N)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child
    
    def get_ucb(self, child, iteration, log_N=None):
        """Compute UCB value with optional simulated annealing."""
        if log_N is None:
            log_N = math.log(self.visit_count)
        
        with child.lock:
            if not child._ucb_dirty and child._cached_ucb is not None:
                return child._cached_ucb
            
            q_value = child.value_sum / child.visit_count
            
            # Simulated annealing UCT
            if self.args.get('use_annealing', False):
                total_searches = self.args.get('num_searches', 1000)
                temperature = self.args['C'] * (1 - iteration / total_searches)
                exploration = temperature * math.sqrt(log_N / child.visit_count)
            else:
                exploration = self.args['C'] * math.sqrt(log_N / child.visit_count)
            
            ucb = q_value + exploration
            child._cached_ucb = ucb
            child._ucb_dirty = False
            return ucb
    
    def expand(self):
        """Expand node by adding a new child."""
        valid_indices = np.where(self.valid_moves == 1)[0]
        action = np.random.choice(valid_indices)
        self.valid_moves[action] = 0
        
        if np.sum(self.valid_moves) == 0:
            self.is_full = True
        
        child_state = self.state.copy()
        child_state = self.env.get_next_state(child_state, action)

        child = AdvancedNode(self.env, self.args, child_state, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        """Perform rollout simulation."""
        tmp = self.state.copy()
        if self.args.get('simulate_with_priority', False):
            return simulate_with_priority_nb(
                tmp, self.env.row_count, self.env.column_count,
                self.env.pts_upper_bound, self.env.priority_grid,
                self.args.get('top_n', 1)
            )
        else:
            return simulate_nb(
                tmp, self.env.row_count, self.env.column_count,
                self.env.pts_upper_bound
            )
    
    def backpropagate(self, value):
        """Backpropagate value up the tree."""
        with self.lock:
            self.value_sum += value
            self._ucb_dirty = True
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

# =============================================================================
# Unified MCTS Classes
# =============================================================================

class UnifiedMCTS:
    """Unified MCTS implementation with all advanced features."""
    
    def __init__(self, env, args, variant='basic'):
        self.env = env  # The N3ilUnified game environment
        self.args = args
        self.variant = variant
        
        # Configure variant-specific settings
        if variant == 'basic':
            self.args['simulate_with_priority'] = False
            self.args['use_annealing'] = False
            self.args['num_workers'] = 1
        elif variant == 'priority':
            self.args['simulate_with_priority'] = True
            self.args['use_annealing'] = False
            self.args['num_workers'] = 1
        elif variant == 'parallel':
            self.args['simulate_with_priority'] = False
            self.args['use_annealing'] = False
            self.args['num_workers'] = self.args.get('num_workers', 4)
        elif variant == 'advanced':
            self.args['simulate_with_priority'] = True
            self.args['use_annealing'] = True
            self.args['num_workers'] = self.args.get('num_workers', 4)
    
    def search(self, state):
        """Perform MCTS search from given state."""
        # Use parallel MCTS for variants that support it
        if self.variant in ['parallel', 'advanced'] and self.args.get('num_workers', 1) > 1:
            return self._parallel_search(state)
        
        root = AdvancedNode(self.env, self.args, state)
        
        if self.args.get('process_bar', False):
            search_iterator = trange(self.args['num_searches'])
        else:
            search_iterator = range(self.args['num_searches'])
        
        for iteration in search_iterator:
            node = root
            
            # Selection
            while node.is_fully_expanded():
                node = node.select(iteration)
            
            # Expansion and simulation
            if node.action_taken is not None:
                value, is_terminal = self.env.get_value_and_terminated(
                    node.state, node.valid_moves
                )
                if not is_terminal:
                    node = node.expand()
                    value = node.simulate()
            else:
                node = node.expand()
                value = node.simulate()
            
            # Backpropagation
            node.backpropagate(value)
        
        # Convert visit counts to action probabilities
        action_probs = np.zeros(self.env.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
    def _parallel_search(self, state):
        """Parallel MCTS search implementation."""
        root = AdvancedNode(self.env, self.args, state)
        num_workers = self.args.get('num_workers', 4)
        
        sims_per_worker = self.args['num_searches'] // num_workers
        remainder = self.args['num_searches'] % num_workers
        
        def worker(n_sims):
            for i in range(n_sims):
                self._search_once(root, i * num_workers)
        
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(worker, sims_per_worker) for _ in range(num_workers)]
            if remainder:
                futures.append(pool.submit(worker, remainder))
            wait(futures)
        
        # Convert to action probabilities
        action_probs = np.zeros(self.env.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
    def _search_once(self, root, iteration):
        """Single MCTS simulation with virtual loss."""
        path = []
        node = root
        
        # Selection with virtual loss
        while node.is_fully_expanded():
            path.append(node)
            node.apply_virtual_loss()
            node = node.select(iteration)
        
        # Expansion/Simulation
        if node.action_taken is None:
            node = node.expand()
        path.append(node)
        node.apply_virtual_loss()
        
        value = node.simulate() if not self.env.get_value_and_terminated(
            node.state, node.valid_moves
        )[1] else self.env.get_value_and_terminated(
            node.state, node.valid_moves
        )[0]
        
        # Revert virtual loss and backpropagate
        for n in path:
            n.revert_virtual_loss()
        node.backpropagate(value)

class LegacyUnifiedParallelMCTS(UnifiedMCTS):
    """Legacy parallel MCTS - now integrated into UnifiedMCTS with variant='parallel'."""
    
    def __init__(self, game, args):
        super().__init__(game, args, variant='parallel')
        print("Warning: LegacyUnifiedParallelMCTS is deprecated. Use UnifiedMCTS with variant='parallel'.")
    
    # Legacy methods - now handled by base UnifiedMCTS
    def _search_once(self, root, iteration):
        return super()._search_once(root, iteration)
    
    def search(self, state):
        return super().search(state)

# =============================================================================
# Action Selection Strategies
# =============================================================================

def select_outermost_with_tiebreaker(mcts_probs, n):
    """Select action from outermost positions among highest probability."""
    mcts_probs_2d = mcts_probs.reshape((n, n))
    max_val = np.max(mcts_probs_2d)
    max_indices = np.argwhere(mcts_probs_2d == max_val)
    
    def edge_distance(i, j):
        return min(i, n - 1 - i, j, n - 1 - j)
    
    distances = [edge_distance(i, j) for i, j in max_indices]
    min_dist = min(distances)
    
    outermost_positions = [pos for pos, dist in zip(max_indices, distances) 
                          if dist == min_dist]
    
    chosen_pos = outermost_positions[np.random.choice(len(outermost_positions))]
    action = chosen_pos[0] * n + chosen_pos[1]
    return action

# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_unified(args, variant='basic'):
    """Evaluate unified MCTS implementation."""
    # Setup priority system
    if args.get('priority_type') == 'supnorm':
        priority_system = SupNormPriority()
    else:
        priority_system = CollinearCountPriority()
    
    # Create game environment
    n = args['n']
    game = N3ilUnified(grid_size=(n, n), args=args, priority_system=priority_system)
    
    # Create MCTS instance with variant
    mcts = UnifiedMCTS(game, args, variant=variant)
    
    # Run evaluation
    start_time = time.time()
    state = game.get_initial_state()
    num_points = 0
    
    while True:
        if args.get('display_state', False):
            print(f"Points: {num_points}")
            print(state)
        
        valid_moves = game.get_valid_moves(state)
        value, is_terminal = game.get_value_and_terminated(state, valid_moves)
        
        if is_terminal:
            end_time = time.time()
            if args.get('display_state', False):
                print(f"Terminated with {num_points} points in {end_time - start_time:.2f}s")
            break
        
        # Get MCTS action probabilities
        mcts_probs = mcts.search(state)
        action = select_outermost_with_tiebreaker(mcts_probs, n)
        
        # Apply action
        num_points += 1
        state = game.get_next_state(state, action)
    
    return num_points if args.get('logging_mode', False) else None
