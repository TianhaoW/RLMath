import itertools
import numpy as np
print(np.__version__)
# np.random.seed(0)  # Removed global seed, will be set per experiment
from tqdm import trange
from numba import njit
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, as_completed
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time
import datetime
import re
from collections import defaultdict
import pprint
import math
from typing import Tuple, List, Set, Callable, NamedTuple, Union, Optional, Iterable, Dict
from multiprocessing import Pool
from sympy import Rational, Integer
from sympy.core.numbers import igcd
from src.envs import N3il, N3il_with_symmetry, supnorm_priority, supnorm_priority_array

import psutil
import os

def set_seeds(seed):
    """Set random seeds for reproducibility across all random number generators."""
    np.random.seed(seed)
    random.seed(seed)
    # Force compilation of numba functions with the seeded state
    # This ensures numba's internal random state is also seeded
    from numba import config
    config.THREADING_LAYER = 'safe'

@njit(cache=True, nogil=True)
def exploration_decay_nb(x):  # Monotone-down from (0,1) to (1,0)
    # Linear
    # return 1 - 0.7 * x   # Found optimal 4-point solution: 86/100 times (86.0%)
    # return 1 - x # 85/100 times (85.0%)

    # Square root (gentle early decay)
    # return 1 - 0.9 * np.sqrt(x) # 91/100 times (91.0%)
    # return 1 - 1 * np.sqrt(x) # 83/100 times (83.0%)
    # return 1 - 0.5 * np.sqrt(x) # 88/100 times (88.0%)
    # return 1 - 0.7 * np.sqrt(x) # 86%
    #return 1 - 0.8 * np.sqrt(x) # 92/100 times (92.0%)
    return 1 - 0.85 * np.sqrt(x)

    # Quadratic (faster decay)
    # return 1 - (x ** 2)

    # Exponential (custom normalization)
    # return ((np.exp(1)/(np.exp(1)-1))**2) * ((np.exp(-x)-np.exp(-1)) ** 2) # 85/100 times (85.0%)

    # Exponential fast (k=3)
    #k = 3.0
    # return (np.exp(-k * x) - np.exp(-k)) / (1 - np.exp(-k)) # 90/100 times (90.0%)

    # Exponential slow (k=1)
    # k = 1.0
    # return (np.exp(-k * x) - np.exp(-k)) / (1 - np.exp(-k)) # 86/100 times (86.0%)

    # Cosine decay
    # return 0.5 * (1 + np.cos(np.pi * x)) # 85/100 times (85.0%)

    # Rational decay
    # a = 1.0
    # return (1 - x) / (1 + a * x) # solution: 90/100 times (90.0%)

    # Logistic decay
    # k = 10.0
    # g0 = 1 / (1 + np.exp(k * (0 - 0.5)))
    # g1 = 1 / (1 + np.exp(k * (1 - 0.5)))
    # gx = 1 / (1 + np.exp(k * (x - 0.5)))
    # return (gx - g1) / (g0 - g1) # 86/100 times (86.0%)

    # Cubic decay
    # return 1 - x ** 3 # 91/100 times (91.0%)
    # return 1 - (0.9 * (x ** 3)) # 91/100 times (91.0%)

@njit(cache=True, nogil=True)
def value_fn_nb(x):
    # return x
    # return np.exp(x)
    return x

@njit(cache=True, nogil=True)
def get_value_nb(state, pts_upper_bound, value_f=value_fn_nb):
    total = np.sum(state)
    n = pts_upper_bound/2
    
    # === REVERSE REWARDING FUNCTIONS (prefer smaller point counts) ===
    
    # 1. Simple Linear Inverse: 1.0 for empty board, 0.0 for full board
    # return (n - total) / n  # Range: [0, 1]
    
    # 2. Exponential Decay (Strong preference for fewer points)
    # return np.exp(-2.0 * (total / n))  # Range: [e^-2, 1] ≈ [0.135, 1]
    # return np.exp(-1.0 * (total / n))  # Range: [e^-1, 1] ≈ [0.368, 1]
    # return np.exp(-0.5 * (total / n))  # Range: [e^-0.5, 1] ≈ [0.607, 1]
    
    # 3. Power Functions (Adjustable curvature)
    # return ((n - total) / n) ** 2  # Quadratic preference: [0, 1]
    # return ((n - total) / n) ** 0.5  # Square root preference: [0, 1]
    # return ((n - total) / n) ** 3  # Cubic preference (very aggressive): [0, 1]
    
    # 4. Sigmoid-based (Smooth transition around target)
    # target = n * 0.3  # Target 30% of grid filled
    # return 1.0 / (1.0 + np.exp(0.5 * (total - target)))  # Range: ≈[0, 1]
    # return 1.0 / (1.0 + np.exp(1.0 * (total - target)))  # Steeper transition
    
    # 5. Logarithmic Penalty
    # return max(0, 1.0 - np.log(1.0 + total) / np.log(1.0 + n))  # Range: [0, 1]
    
    # 6. ReLU-based with different thresholds
    # return max(0, (1.2 * n - total) / n)  # Reward up to 120% of n: [0, 1.2]
    # return max(0, (1.5 * n - total) / n)  # Current: reward up to 150% of n
    
    # === OPTIMAL FOR 3x3 MINIMAL COMPLETE SET (4 points) ===
    # Simple linear inverse works best for finding exact minimal sets
    return (1.6*n - total) * n  / (1.6 - 1.3)# Range: [0, 1], 1.0 for empty, 0.0 for full !!!CURRENT OPTIMAL!!!

    # Baseline rewarding function
    '''
    baseline = 1.6 * n
    theoretical_min = 1.3 * n
    num = baseline - total
    if num > 0:
        return num / (baseline - theoretical_min)  # Range: [0, 1], 1.0 for empty, 0.0 for full
    if num <= 0:
        return num / (baseline - theoretical_min)  # Range: [-1, 0], 0.0 for empty, -1.0 for full
    '''
    # Numba-safe scalar casts
    total = np.float64(np.sum(state))
    n = np.float64(pts_upper_bound) / 2.0

    # Target and normalization
    target = 0.9 * n
    max_possible = 2.0 * n
    eps = np.float64(1e-12)
    span = np.maximum(max_possible - target, eps)  # avoid division by zero
    # Normalized distance: 0 at target, 1 at 2n (can be < 0 if total < target)
    tnorm = (total - target) / span

    # ---- Choose ONE of the following returns (uncomment exactly one) ----

    # 2) Quadratic (penalizes farther from target more strongly)
    # return np.clip(1.0 - tnorm * tnorm, 0.0, 1.0)

    # 3) Gaussian peak at target (default active; sharp pull to 0.9n)
    # sigma = np.maximum(0.05 * n, eps)  # controls sharpness
    # return np.exp(-0.5 * ((total - target) / sigma) ** 2)

    # 4) Logistic decay from target upward
    # k = 6.0 / np.maximum(n, 1.0)
    # return 1.0 / (1.0 + np.exp(k * (total - target)))

    # 5) Rational distance penalty (gentler tail)
    # alpha = 2.0 / np.maximum(n, 1.0)
    # return 1.0 / (1.0 + alpha * np.abs(total - target))

    # 6) Piecewise: full at/below target, then linear drop to 0 at 2n
    # if total <= target:
    #     return 1.0
    # else:
    #     return np.maximum(0.0, 1.0 - (total - target) / span)

    # 7) Cosine half-wave on [target, 2n] (smooth with zero slope at target)
    # x = np.clip(tnorm, 0.0, 1.0)               # map [target,2n] -> [0,1]
    # return 0.5 * (1.0 + np.cos(np.pi * x))     # 1 at target, 0 at 2n

    # ------------ Positive-direction variants (optimum at 2n) ------------
    # Use these if you want to test the opposite objective (larger total better).
    # 1+) Linear increasing from target to 2n
    # return np.clip(tnorm, 0.0, 1.0)

    # 2+) Quadratic increasing (slow start, faster near 2n)
    # x = np.clip(tnorm, 0.0, 1.0)
    # return x * x

    # 3+) Exponential rise (very low until near 2n)
    # x = np.clip(tnorm, 0.0, 1.0)
    # k = 4.0
    # return (np.exp(k * x) - 1.0) / (np.exp(k) - 1.0)

# JIT-compiled function to check if three points are collinear
@njit(cache=True, nogil=True)
def _are_collinear(x1, y1, x2, y2, x3, y3):
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

# JIT-compiled function to determine valid moves on the board 
@njit(cache=True, nogil=True)
def get_valid_moves_nb(state, row_count, column_count):
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
    """
    Given a parent state (2D boolean array) and its valid move mask (1D uint8 array),
    return a refined valid move mask for the child:
      1) Remove the action just taken.
      2) For each existing point in state, compute the line to the new point,
         then invalidate any intermediate grid points that lie exactly on that line.
      3) If slope is infinite, invalidate entire column; if slope is zero, invalidate entire row.
    Returns a flattened uint8 array: 1 = valid, 0 = invalid.
    """
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

            # General (non-vertical, non-horizontal) case: remove every point on the infinite line
            # through (new_r,new_c) and (pr,pc), including both the segment and its extensions.
            for cc in range(column_count):
                # compute how far horizontally from the new point
                num = (cc - new_c) * dr
                # only those aligning to integer row are collinear
                if num % dc != 0:
                    continue
                rr = new_r + num // dc
                # skip anything outside the grid
                if rr < 0 or rr >= row_count:
                    continue
                idx = rr * column_count + cc
                mask[idx] = 0

    return mask

# JIT-compiled function to count collinear triples on the board
@njit(cache=True, nogil=True)
def check_collinear_nb(state, row_count, column_count):
    max_pts = row_count * column_count
    coords = np.empty((max_pts, 2), np.int64)
    n_pts = 0

    # Collect all placed point coordinates
    for i in range(row_count):
        for j in range(column_count):
            if state[i, j] == 1:
                coords[n_pts, 0] = i
                coords[n_pts, 1] = j
                n_pts += 1

    triples = 0
    # Count all collinear triplets
    for a in range(n_pts):
        for b in range(a + 1, n_pts):
            for c in range(b + 1, n_pts):
                i1, j1 = coords[a, 0], coords[a, 1]
                i2, j2 = coords[b, 0], coords[b, 1]
                i3, j3 = coords[c, 0], coords[c, 1]
                if _are_collinear(j1, i1, j2, i2, j3, i3):
                    triples += 1
    return triples

@njit(cache=True, nogil=True)
def simulate_nb(state, row_count, column_count, pts_upper_bound):
    """
    Perform random rollout until no valid moves remain.
    Return normalized value using a custom value function.
    Uses get_valid_moves_subset_nb for incremental validity updates.
    Note: This function uses numba's random number generator which is seeded globally.
    """
    max_size = row_count * column_count
    # Initial valid moves mask
    valid_moves = get_valid_moves_nb(state, row_count, column_count)
    total_valid = np.sum(valid_moves)

    while total_valid > 0:
        # Build list of valid actions
        acts = np.empty(total_valid, np.int64)
        k = 0
        for idx in range(max_size):
            if valid_moves[idx]:
                acts[k] = idx
                k += 1
        # Randomly select one valid action and place the point
        pick = acts[np.random.randint(0, total_valid)]

        # Incrementally update valid_moves using subset-based filtering
        valid_moves = get_valid_moves_subset_nb(
            state,
            valid_moves,
            pick,
            row_count,
            column_count
        )

        r = pick // column_count
        c = pick % column_count
        state[r, c] = 1  # mark the new point

        total_valid = np.sum(valid_moves)

    # Compute and return the final value
    return get_value_nb(state, pts_upper_bound)

@njit(cache=True, nogil=True)
def filter_top_priority_moves(valid_moves, priority_grid, row_count, column_count, top_N=1):
    """
    Numba-accelerated: Filter valid moves to only those with the top_N highest priorities.

    Args:
        valid_moves (np.ndarray): 1D array (flattened) of valid moves (1=valid, 0=invalid).
        priority_grid (np.ndarray): 2D array of priority values for each grid cell.
        row_count (int): Number of rows in the grid.
        column_count (int): Number of columns in the grid.
        top_N (int): Number of top priority levels to select.

    Returns:
        np.ndarray: 1D mask array with only the top_N-priority valid moves set to 1.
    """
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

    # Find the unique priorities and sort descending
    # Numba doesn't support np.unique or sort for lists, so do it manually
    # 1. Copy priorities to a new array
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
    # 2. Sort unique_priorities descending (simple selection sort)
    for i in range(len(unique_priorities)):
        max_idx = i
        for j in range(i+1, len(unique_priorities)):
            if unique_priorities[j] > unique_priorities[max_idx]:
                max_idx = j
        # Swap
        tmp = unique_priorities[i]
        unique_priorities[i] = unique_priorities[max_idx]
        unique_priorities[max_idx] = tmp

    # 3. Select top_N priorities
    N = min(top_N, len(unique_priorities))
    threshold = unique_priorities[:N]

    # 4. Build mask
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
def simulate_with_priority_nb(state, row_count, column_count, pts_upper_bound, priority_grid, top_N):
    """
    Perform a random rollout that first filters valid moves by priority
    and then proceeds like simulate_nb, but initial valid moves are pre-filtered.
    Args:
        state (np.ndarray): 2D board state.
        row_count (int): Number of rows.
        column_count (int): Number of columns.
        pts_upper_bound (int): Scoring upper bound.
        priority_grid (np.ndarray): 2D array of priorities.
        top_N (int): Number of top priority levels to keep.
    Returns:
        float: Normalized final value.
    """
    max_size = row_count * column_count

    # Initial valid moves mask
    valid_moves = get_valid_moves_nb(state, row_count, column_count)
    # Pre-filter by priority
    valid_moves = filter_top_priority_moves(
        valid_moves, priority_grid, row_count, column_count, top_N
    )
    total_valid = np.sum(valid_moves)

    # Rollout until no moves remain
    while total_valid > 0:
        acts = np.empty(total_valid, np.int64)
        k = 0
        for idx in range(max_size):
            if valid_moves[idx]:
                acts[k] = idx
                k += 1

        pick = acts[np.random.randint(0, total_valid)]

        # Update valid moves and state
        valid_moves = get_valid_moves_subset_nb(
            state, valid_moves, pick, row_count, column_count
        )
        state[pick // column_count, pick % column_count] = 1

        # Filter again by priority
        valid_moves = filter_top_priority_moves(
            valid_moves, priority_grid, row_count, column_count, top_N
        )
        total_valid = np.sum(valid_moves)

    return get_value_nb(state, pts_upper_bound)

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)

        if parent is None:
            self.level = np.sum(state)  # Level is the number of points placed
            if self.level <= game.max_level_to_use_symmetry:
                self.valid_moves = game.get_valid_moves_with_symmetry(state)
            else:
                self.valid_moves = game.get_valid_moves(state)
        else:
            self.level = parent.level + 1
            if self.level <= game.max_level_to_use_symmetry:
                self.valid_moves = game.get_valid_moves_subset_with_symmetry(
                    parent.state, parent.valid_moves, self.action_taken)
            else:
                self.valid_moves = game.get_valid_moves_subset(
                    parent.state, parent.valid_moves, self.action_taken)



        self.is_full = False
        self._cached_ucb = None     # Cached UCB value
        self._ucb_dirty = True      # Indicates whether the cached UCB is stale

    def apply_virtual_loss(self):
        with self.lock:
            self.value_sum -= self._vl
            self.visit_count += 1
            self._ucb_dirty = True  # Mark UCB as outdated

    def revert_virtual_loss(self):
        with self.lock:
            self.value_sum += self._vl
            self._ucb_dirty = True  # Mark UCB as outdated

    def is_fully_expanded(self):
        return self.is_full and len(self.children) > 0

    def select(self, iter):
        best_child = None
        best_ucb = -np.inf
        log_N = math.log(self.visit_count)

        for child in self.children:
            ucb = self.get_ucb(child, iter, log_N)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, iter, log_N=None):
        if log_N is None:
            log_N = math.log(self.visit_count)

        with child.lock:
            if not child._ucb_dirty and child._cached_ucb is not None:
                return child._cached_ucb

            q_value = child.value_sum / child.visit_count
            T_i = self.args['C'] * exploration_decay_nb(iter/self.args['num_searches'])
            exploration_value = T_i * math.sqrt(log_N / child.visit_count)
            ucb = q_value + exploration_value
            # print("Exploit:", q_value)
            # print("Explore:", exploration_value)
            child._cached_ucb = ucb
            child._ucb_dirty = False
            return ucb

    def expand(self):
        valid_indices = np.where(self.valid_moves == 1)[0]
        action = np.random.choice(valid_indices)
        self.valid_moves[action] = 0

        if np.sum(self.valid_moves) == 0:
            self.is_full = True

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        tmp = self.state.copy()
        if self.args["simulate_with_priority"] == True:
            return simulate_with_priority_nb(tmp,
                                            self.game.row_count,
                                            self.game.column_count,
                                            self.game.pts_upper_bound,
                                            self.game.priority_grid,
                                            self.args['TopN'])
        else:
            return simulate_nb(tmp,
                            self.game.row_count,
                            self.game.column_count,
                            self.game.pts_upper_bound)

    def backpropagate(self, value):
        with self.lock:
            self.value_sum += value
            self._ucb_dirty = True  # Mark UCB as outdated
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, game, args={
        'num_searches': 1000,
        'C': 1.4
    }):
        self.game = game
        self.args = args

    def search(self, state):
        # define root
        root = Node(self.game, self.args, state)

        if self.args['process_bar'] == True:
            search_iterator = trange(self.args['num_searches'])
        else:
            search_iterator = range(self.args['num_searches'])

        for search in search_iterator:
            node = root

            # selection
            while node.is_fully_expanded():
                node = node.select(iter=search)

            if node.action_taken is not None:
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.valid_moves)
                # has_collinear = self.game.check_collinear(node.state, node.action_taken)
                # value, _ = self.game.get_value_and_terminated(node.state)

                if not is_terminal:
                    node = node.expand()
                    value = node.simulate()
            else:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
            
            # expansion
            # simulation
            # backpropagation

        # return visit_counts

class ParallelMCTS(MCTS):
    def __init__(self, game, args):
        super().__init__(game, args)
        self.num_workers   = args.get('num_workers', 4)
        self.virtual_loss  = args.get('virtual_loss', 1.0)
        self.args = args

    # --- single simulation --------------------------------------------------
    def _search_once(self, root, worker_iter):
        path = []
        node = root

        # 1. SELECTION
        while node.is_fully_expanded():
            path.append(node)
            node.apply_virtual_loss()         # <‑‑ reserve
            node = node.select(iter=worker_iter*self.num_workers)

        # 2. EXPANSION / SIMULATION
        if node.action_taken is None:
            node = node.expand()
        path.append(node)
        node.apply_virtual_loss()             # reserve leaf
        value = node.simulate() if not self.game.get_value_and_terminated(
            node.state, node.valid_moves
        )[1] else self.game.get_value_and_terminated(
            node.state, node.valid_moves
        )[0]

        # 3. UNDO VIRTUAL LOSS + BACKPROP
        for n in path:
            n.revert_virtual_loss()
        node.backpropagate(value)
    # ------------------------------------------------------------------------

    # --- parallel driver ----------------------------------------------------
    def search(self, state):
        root = Node(self.game, self.args, state)

        sims_per_worker = self.args['num_searches'] // self.num_workers
        remainder       = self.args['num_searches'] %  self.num_workers

        def worker(n_sims, worker_id=0):
            # Set deterministic seed for this worker thread
            if 'random_seed' in self.args:
                worker_seed = self.args['random_seed'] + worker_id * 10000
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            
            if self.args['process_bar'] == True:
                for worker_iter in trange(n_sims):
                    self._search_once(root, worker_iter)
            else:
                for worker_iter in range(n_sims):
                    self._search_once(root, worker_iter)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(worker, sims_per_worker, worker_id)
                       for worker_id in range(self.num_workers)]
            if remainder:                     # handle leftovers
                futures.append(pool.submit(worker, remainder, self.num_workers))
            wait(futures)

        # convert visit counts → prob. vector
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    

# ---------- Leaf/Child Paralllel MCTS ------------------------------------------------

def _rollout_many(child, R: int):
    """Run child.simulate() R times in the same worker thread; return the list of values."""
    if R <= 1:
        return [child.simulate()]
    vals = []
    for _ in range(R):
        vals.append(child.simulate())
    return vals

# --- Child-parallel expansion node ---
class LeafChildParallelNode(Node):
    """
    Node class that expands all children in parallel (child-parallel expansion).
    """
    def __init__(self, game, args, state, parent=None, action_taken=None):
        super().__init__(game, args, state, parent, action_taken)

    def expand(self):
        """
        Expand all valid children in parallel and return a randomly chosen child.
        Important: do not mutate self.valid_moves until children are constructed,
        because Node.__init__ of children depends on parent's valid mask.
        """
        # Snapshot the parent's valid mask first
        with self.lock:
            valid_indices = np.where(self.valid_moves == 1)[0]
            if len(valid_indices) == 0:
                self.is_full = True
                return None
            parent_valid_snapshot = self.valid_moves.copy()
            parent_state = self.state  # read-only usage below

        # Build children in parallel
        def build_child(action):
            # Construct child state
            child_state = parent_state.copy()
            child_state = self.game.get_next_state(child_state, action)
            # Create child node (this will compute valid moves once)
            child = LeafChildParallelNode(self.game, self.args, child_state, self, action)
            # Optionally, we can override with our own subset computation using the snapshot:
            # This avoids recomputing if Node.__init__ is heavy or if we want to be explicit.
            try:
                child_valid = get_valid_moves_subset_nb(
                    parent_state,
                    parent_valid_snapshot,
                    action,
                    self.game.row_count,
                    self.game.column_count
                )
                child.valid_moves = child_valid
            except Exception:
                # Fallback: keep whatever Node.__init__ computed
                pass
            return child

        children = []
        with ThreadPoolExecutor(max_workers=min(len(valid_indices), 32)) as executor:
            futures = [executor.submit(build_child, a) for a in valid_indices]
            for fut in futures:
                children.append(fut.result())

        # Append children and then mark fully expanded
        with self.lock:
            self.children.extend(children)
            # Now it is safe to mark all these actions as used
            self.valid_moves[valid_indices] = 0
            self.is_full = True

        # Return a random child for compatibility with the base MCTS flow
        return random.choice(children) if children else None

class LeafChildParallelMCTS(MCTS):
    """
    MCTS variant that supports both leaf-parallel and child-parallel simulation strategies.
    - Leaf-parallel: Run multiple rollouts from a leaf before backpropagation.
    - Child-parallel: When expanding a node, run one simulation from each child in parallel.
      (If both are set, runs multiple rollouts per child in parallel.)
    """
    def __init__(self, game, args):
        super().__init__(game, args)
        self.num_workers = args.get('num_workers', 4)
        self.simulations_per_leaf = args.get('simulations_per_leaf', 1)
        self.child_parallel = args.get('child_parallel', True)
        self.virtual_loss = args.get('virtual_loss', 1.0)
        self.args = args

    def _simulate_leaf_parallel(self, node, num_simulations):
        """
        Run multiple rollouts from a single leaf; return list of values.
        No backprop here; caller will backprop in main thread.
        """
        if num_simulations <= 1:
            return [node.simulate()]
        # Usually we don't need virtual loss here, selection is single-threaded
        with ThreadPoolExecutor(max_workers=min(num_simulations, self.num_workers)) as pool:
            futures = [pool.submit(node.simulate) for _ in range(num_simulations)]
            return [f.result() for f in futures]

    def _simulate_child_parallel(self, node, num_simulations):
        """
        Expand all valid children and run multiple rollouts per child in parallel.
        Returns a list of tuples: (child, [values...]).
        """
        # Snapshot parent's valid mask to avoid mutation races
        with node.lock:
            valid_indices = np.where(node.valid_moves == 1)[0]
            if valid_indices.size == 0:
                node.is_full = True
                return []
            parent_valid_snapshot = node.valid_moves.copy()
            parent_state = node.state

        # Build children in parallel first (no backprop here)
        def build_child(action):
            child_state = parent_state.copy()
            child_state = self.game.get_next_state(child_state, action)
            child = LeafChildParallelNode(self.game, self.args, child_state, node, action)
            # Override child's valid mask using snapshot so we don't recompute later
            try:
                child_valid = get_valid_moves_subset_nb(
                    parent_state,
                    parent_valid_snapshot,
                    action,
                    self.game.row_count,
                    self.game.column_count
                )
                child.valid_moves = child_valid
            except Exception:
                pass
            return child

        children = []
        with ThreadPoolExecutor(max_workers=min(len(valid_indices), self.num_workers)) as pool:
            futures = [pool.submit(build_child, a) for a in valid_indices]
            for fut in futures:
                children.append(fut.result())

        # Attach children and mark node fully expanded
        with node.lock:
            node.children.extend(children)
            node.valid_moves[valid_indices] = 0
            node.is_full = True

        # Now run R rollouts per child in parallel; collect results (no backprop here)
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futs = [pool.submit(_rollout_many, child, num_simulations) for child in children]
            for child, fut in zip(children, futs):
                vals = fut.result()
                results.append((child, vals))

        # Backprop only in the main thread to avoid data races
        for child, vals in results:
            for v in vals:
                child.backpropagate(v)

        return results

    def search(self, state):
        root = LeafChildParallelNode(self.game, self.args, state)
        num_searches = self.args.get('num_searches', 1000)
        process_bar = self.args.get('process_bar', False)
        search_iterator = trange(num_searches) if process_bar else range(num_searches)

        for search in search_iterator:
            node = root
            # 1) Selection (single-threaded)
            while node.is_fully_expanded():
                node = node.select(iter=search)

            # 2) If child-parallel enabled and node not yet expanded → expand + run
            if self.child_parallel and not node.is_fully_expanded():
                self._simulate_child_parallel(node, self.simulations_per_leaf)
                # Everything done inside; continue to next simulation
                continue

            # 3) Otherwise do leaf-parallel on a single child
            # If node is not root and terminal, just backprop its terminal value
            if node.action_taken is not None:
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.valid_moves)
                if is_terminal:
                    node.backpropagate(value)
                    continue
                # Expand one child (this expand returns a random child)
                node = node.expand()
                if node is None:
                    # No child could be expanded; treat as terminal with zero
                    continue
                values = self._simulate_leaf_parallel(node, self.simulations_per_leaf)
            else:
                # Root case: expand first
                node = node.expand()
                if node is None:
                    continue
                values = self._simulate_leaf_parallel(node, self.simulations_per_leaf)

            # Backprop the leaf-parallel values
            for v in values:
                node.backpropagate(v)

        # 4) Action probabilities from root's children
        action_probs = np.zeros(self.game.action_size)
        total = 0
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
            total += child.visit_count
        if total > 0:
            action_probs /= total
        return action_probs


class MCGSNode:
    """Node class for Monte Carlo Graph Search"""
    def __init__(self, state, state_key):
        self.state = state
        self.state_key = state_key
        self.lower_bound = 0.0
        self.upper_bound = 1.0  # V_max equivalent
        self.outgoing_edges = {}  # action -> (reward, next_state_key)
        self.has_outgoing_edges = False

class MCGS:
    """Monte Carlo Graph Search implementation compatible with MCTS interface"""
    def __init__(self, game, args={
        'num_searches': 1000,
        'C': 1.4,
        'gamma': 0.99  # discount factor for MCGS
    }):
        self.game = game
        self.args = args
        self.gamma = args.get('gamma', 0.99)
        self.V_max = 1.0 / (1.0 - self.gamma)
        self.graph = {}  # state_key -> MCGSNode
        
    def search(self, state):
        """Main MCGS algorithm following the pseudocode"""
        # Initialize graph with root node
        root_key = self.game.state_to_key(state)
        if root_key not in self.graph:
            self.graph[root_key] = MCGSNode(state.copy(), root_key)
        
        budget = self.args.get('num_searches', 1000)
        
        if self.args.get('process_bar', False):
            search_iterator = trange(budget)
        else:
            search_iterator = range(budget)
            
        for n in search_iterator:
            # Step 1: Bellman backups to compute value bounds
            self._compute_value_bounds()
            
            # Step 2: Optimistic sampling - follow path with highest upper bounds
            leaf_key = self._optimistic_sampling(root_key)
            
            # Step 3: Node expansion - add all possible actions from leaf
            if leaf_key in self.graph:
                self._expand_node(leaf_key)
        
        # Step 4: Return action probabilities based on lower bounds
        return self._get_action_probabilities(root_key)
    
    def _compute_value_bounds(self):
        """Compute lower and upper value bounds using Bellman operators"""
        # Initialize bounds
        for node in self.graph.values():
            if not node.has_outgoing_edges:
                # Sink nodes: evaluate using the game's evaluation function
                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, self.game.get_valid_moves(node.state)
                )
                if is_terminal:
                    node.lower_bound = value
                    node.upper_bound = value
                else:
                    node.lower_bound = 0.0
                    node.upper_bound = self.V_max
        
        # Iterate Bellman operators until convergence
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            old_lower = {k: v.lower_bound for k, v in self.graph.items()}
            old_upper = {k: v.upper_bound for k, v in self.graph.items()}
            
            # Update internal nodes
            for node in self.graph.values():
                if node.has_outgoing_edges:
                    # Lower bound update
                    max_lower = -np.inf
                    max_upper = -np.inf
                    
                    for action, (reward, next_key) in node.outgoing_edges.items():
                        if next_key in self.graph:
                            next_node = self.graph[next_key]
                            q_lower = reward + self.gamma * next_node.lower_bound
                            q_upper = reward + self.gamma * next_node.upper_bound
                            max_lower = max(max_lower, q_lower)
                            max_upper = max(max_upper, q_upper)
                    
                    node.lower_bound = max_lower if max_lower > -np.inf else 0.0
                    node.upper_bound = max_upper if max_upper > -np.inf else self.V_max
            
            # Check convergence
            converged = True
            for key in self.graph:
                if (abs(self.graph[key].lower_bound - old_lower[key]) > tolerance or
                    abs(self.graph[key].upper_bound - old_upper[key]) > tolerance):
                    converged = False
                    break
            
            if converged:
                break
    
    def _optimistic_sampling(self, start_key):
        """Follow optimistic policy to reach a leaf node"""
        current_key = start_key
        
        while current_key in self.graph and self.graph[current_key].has_outgoing_edges:
            node = self.graph[current_key]
            
            # Select action with highest upper bound
            best_action = None
            best_value = -np.inf
            
            for action, (reward, next_key) in node.outgoing_edges.items():
                if next_key in self.graph:
                    next_node = self.graph[next_key]
                    q_value = reward + self.gamma * next_node.upper_bound
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
            
            if best_action is not None:
                _, current_key = node.outgoing_edges[best_action]
            else:
                break
        
        return current_key
    
    def _expand_node(self, node_key):
        """Expand a leaf node by trying all possible actions"""
        if node_key not in self.graph:
            return
            
        node = self.graph[node_key]
        valid_moves = self.game.get_valid_moves(node.state)
        valid_actions = np.where(valid_moves == 1)[0]
        
        if len(valid_actions) == 0:
            return
        
        # Try all valid actions
        for action in valid_actions:
            # Generate next state using the game model
            next_state = node.state.copy()
            next_state = self.game.get_next_state(next_state, action)
            next_key = self.game.state_to_key(next_state)
            
            # Compute immediate reward (difference in game value)
            old_value, _ = self.game.get_value_and_terminated(
                node.state, self.game.get_valid_moves(node.state)
            )
            new_value, _ = self.game.get_value_and_terminated(
                next_state, self.game.get_valid_moves(next_state)
            )
            reward = new_value - old_value
            
            # Add edge to graph
            node.outgoing_edges[action] = (reward, next_key)
            
            # Add next state to graph if not already present
            if next_key not in self.graph:
                self.graph[next_key] = MCGSNode(next_state, next_key)
        
        node.has_outgoing_edges = True
    
    def _get_action_probabilities(self, root_key):
        """Convert lower bound Q-values to action probabilities"""
        action_probs = np.zeros(self.game.action_size)
        
        if root_key not in self.graph:
            return action_probs
        
        root_node = self.graph[root_key]
        
        if not root_node.has_outgoing_edges:
            # If no outgoing edges, return uniform over valid moves
            valid_moves = self.game.get_valid_moves(root_node.state)
            valid_actions = np.where(valid_moves == 1)[0]
            if len(valid_actions) > 0:
                for action in valid_actions:
                    action_probs[action] = 1.0 / len(valid_actions)
            return action_probs
        
        # Compute Q-values using lower bounds (conservative estimate)
        q_values = {}
        for action, (reward, next_key) in root_node.outgoing_edges.items():
            if next_key in self.graph:
                next_node = self.graph[next_key]
                q_values[action] = reward + self.gamma * next_node.lower_bound
            else:
                q_values[action] = reward
        
        if not q_values:
            return action_probs
        
        # Convert to probabilities (softmax-like but focused on best actions)
        max_q = max(q_values.values())
        actions_with_max_q = [a for a, q in q_values.items() if abs(q - max_q) < 1e-6]
        
        # Give equal probability to all actions with maximum Q-value
        prob_per_action = 1.0 / len(actions_with_max_q)
        for action in actions_with_max_q:
            action_probs[action] = prob_per_action
        
        return action_probs

def select_outermost_with_tiebreaker(mcts_probs, n):
    """
    Select an action from the outermost positions among those with the highest MCTS probability.
    If multiple actions have the same max probability and distance to edge, break ties randomly.
    Note: Uses numpy's random number generator which should be seeded for reproducibility.
    """
    # Reshape the 1D probability array to 2D grid
    mcts_probs_2d = mcts_probs.reshape((n, n))
    max_val = np.max(mcts_probs_2d)

    # Find all positions with maximum probability
    max_indices = np.argwhere(mcts_probs_2d == max_val)

    # Define distance to nearest board edge
    def edge_distance(i, j):
        return min(i, n - 1 - i, j, n - 1 - j)

    # Compute edge distance for each candidate
    distances = [edge_distance(i, j) for i, j in max_indices]
    min_dist = min(distances)

    # Select all actions with minimum edge distance
    outermost_positions = [pos for pos, dist in zip(max_indices, distances) if dist == min_dist]

    # Break ties randomly among outermost positions
    chosen_pos = outermost_positions[np.random.choice(len(outermost_positions))]
    action = chosen_pos[0] * n + chosen_pos[1]
    return action

def evaluate(args):
    # Set random seeds for reproducibility at the start of evaluation
    if 'random_seed' in args:
        set_seeds(args['random_seed'])
        # Also warmup numba functions with seeded state
        dummy_state = np.zeros((2, 2), dtype=np.int8)
        _ = simulate_nb(dummy_state, 2, 2, 4)
    
    priority_grid_arr = supnorm_priority_array(args['n'])
    start = time.time()
    n = args['n']

    # Define the environment based on args
    if args['environment'] == 'N3il_with_symmetry':
        n3il =  N3il_with_symmetry(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    elif args['environment'] == 'N3il':
        n3il = N3il(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    else:
        raise ValueError(f"Unknown environment: {args['environment']}")

    if args['algorithm'] == 'MCGS':
        if args.get('num_workers', 1) == 1:
            mcts_cls = MCGS
        else:
            raise ValueError("MCGS does not support parallel execution yet.")
    elif args['algorithm'] == 'MCTS':
        # Check if leaf/child parallel is requested
        if args.get('child_parallel', False) or args.get('simulations_per_leaf', 1) > 1:
            mcts_cls = LeafChildParallelMCTS
        elif args.get('num_workers', 1) <= 1:
            mcts_cls = MCTS
        else:
            mcts_cls = ParallelMCTS
    elif args['algorithm'] == 'LeafChildParallelMCTS':
        mcts_cls = LeafChildParallelMCTS
    else:
        raise ValueError(f"Unknown algorithm: {args['algorithm']}")
    
    # Initialize MCTS or MCGS
    mcts = mcts_cls(n3il, args=args)

    state = n3il.get_initial_state()
    num_of_points = 0

    while True:
        if args['display_state'] == True:
            print("---------------------------")
            print(f"Number of points: {num_of_points}")
            print(state)

        valid_moves = n3il.get_valid_moves(state)
        value, is_terminal = n3il.get_value_and_terminated(state, valid_moves)

        if is_terminal:
            print("*******************************************************************")
            print(f"Trial Terminated with {num_of_points} points. Final valid configuration:")
            print(state)
            n3il.display_state(state, mcts_probs)
            end = time.time()
            print(f"Time: {end - start:.6f} sec")
            
            # Record results to table
            n3il.record_to_table(
                terminal_num_points=num_of_points,
                start_time=start,
                end_time=end,
                time_used=end - start
            )
            
            break

        # Get MCTS probabilities
        mcts_probs = mcts.search(state)

        # Use outermost-priority selector to pick action
        action = select_outermost_with_tiebreaker(mcts_probs, n)

        # Display MCTS probabilities and board
        if args['display_state'] == True:
            n3il.display_state(state, mcts_probs)

        # Apply action
        num_of_points += 1
        state = n3il.get_next_state(state, action)
    
    if args['logging_mode'] == True:
        return num_of_points