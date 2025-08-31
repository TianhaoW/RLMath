from numba import njit
import numpy as np
from .get_valid_moves import get_valid_moves_nb, get_valid_moves_subset_nb
from src.rewards.n3il_rewards import get_value_nb

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