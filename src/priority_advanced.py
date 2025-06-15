"""
Advanced priority calculations with grid storage and caching.
Extracted from luoning's 08_04_MCTS_along_prority_topN_Get_MCTS.ipynb
"""

import os
import numpy as np
from typing import Dict, List, Callable, Optional, Iterable
from multiprocessing import Pool

from .geometry import Point, QQ
from .priority import count_points_on_line, get_possible_slopes
from .utils.cpu_utils import count_idle_cpus, binomial


def _slopes_for_dx_chunk(dx_chunk: List[int], grid_size: int) -> List[QQ]:
    """
    Compute unique slopes for a chunk of dx values within the grid.

    Args:
        dx_chunk (List[int]): A sublist of dx integers to process.
        grid_size (int): The size of the grid.

    Returns:
        List[QQ]: List of unique rational slopes generated from dy/dx pairs in the chunk.
    """
    local_seen = set()
    local_slopes = []
    for dx in dx_chunk:
        for dy in range(1, grid_size):
            s = QQ(dy) / QQ(dx)
            if s not in local_seen:
                local_slopes.append(s)
                local_seen.add(s)
    return local_slopes


def get_possible_slopes_parallel(grid_size: int, idle_cores: int = 0) -> set:
    """
    Generate all possible slopes for lines on a grid of given size using parallel processing.

    Args:
        grid_size (int): The size of the grid.
        idle_cores (int): Number of cores to use for parallelism. If <= 1, runs serially.

    Returns:
        Set[Union[QQ, str]]: Set of unique slopes (rational numbers and 'inf' for vertical lines).
    """
    dx_values = list(range(1, grid_size))

    if idle_cores > 1:
        chunk_size = (len(dx_values) + idle_cores - 1) // idle_cores
        chunks = [dx_values[i:i + chunk_size] for i in range(0, len(dx_values), chunk_size)]
        with Pool(idle_cores) as pool:
            results = pool.starmap(_slopes_for_dx_chunk, [(chunk, grid_size) for chunk in chunks])
    else:
        results = [_slopes_for_dx_chunk(dx_values, grid_size)]

    slopes = {QQ(0)}
    for sublist in results:
        for s in sublist:
            slopes.add(s)
            slopes.add(-s)
    slopes.add('inf')
    return slopes


def point_collinear_count_advanced(p1: Point, grid_size: int) -> int:
    """
    Count the number of collinear triples on that line including that point 
    (not including horizontal and vertical line) using parallel processing.

    Args:
        p1 (Point): The point to check.
        grid_size (int): The size of the grid.

    Returns:
        int: Sum over slopes of binomial(count, 2) for points collinear with p1.
    """
    idle = count_idle_cpus()
    slopes = get_possible_slopes_parallel(grid_size, idle_cores=idle)
    counts = sum([
        binomial(count_points_on_line(p1, slope, grid_size), 2)
        for slope in slopes if slope != 0 and slope != 'inf'
    ])
    return counts


def collinear_count_priority_advanced(n: int, noise: float = 0) -> Callable[[Point], float]:
    """
    Create an advanced priority function based on collinear count.
    
    Args:
        n (int): Grid size
        noise (float): Random noise to add for tie-breaking
        
    Returns:
        Callable[[Point], float]: Priority function
    """
    def priority(point: Point) -> float:
        return -point_collinear_count_advanced(point, n) + noise
    return priority


def priority_grid_advanced(n: int, noise: float = 0) -> np.ndarray:
    """
    Return a 2D numpy array of priority values for each point in an n x n grid using advanced calculation.

    Args:
        n (int): The grid size.
        noise (float): Random noise for tie-breaking.

    Returns:
        np.ndarray: 2D array of priority values.
    """
    priority_fn = collinear_count_priority_advanced(n, noise)
    arr = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            arr[x, y] = priority_fn(Point(x, y))
    return arr


def compute_and_save_priority_grids(priority_grid_fn: Callable[[int], np.ndarray], 
                                   size_list: Optional[List[int]] = None, 
                                   output_dir: str = 'priority_grids') -> None:
    """
    Compute priority grids for each n in size_list (if not already saved),
    and save them as .npy files in the output_dir directory.

    Parameters:
    - priority_grid_fn: A function to generate the priority grid, which takes n as input.
    - size_list: List of n values to compute.
    - output_dir: Directory where .npy files will be saved.
    """
    if size_list is None:
        size_list = [5, 10, 25]
        
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for n in size_list:
        # Check if the file already exists
        filename = os.path.join(output_dir, f'priority_grid_{n}.npy')
        if os.path.exists(filename):
            print(f"priority_grid_{n}.npy already exists. Skipping computation.")
            continue  # Skip this n if the file exists

        # Compute the priority grid for the current n
        grid = priority_grid_fn(n)
        
        # Save the grid as a .npy file
        np.save(filename, grid)
        print(f"Saved priority_grid_{n}.npy")


def load_priority_grid(n: int, input_dir: str = 'priority_grids') -> np.ndarray:
    """
    Load the priority grid numpy array for a given n.

    Parameters:
    - n: The n value of the grid to load.
    - input_dir: Directory containing the .npy files.

    Returns:
    - priority_grid (np.ndarray): The loaded priority grid.
    """
    filename = os.path.join(input_dir, f'priority_grid_{n}.npy')
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    grid = np.load(filename)
    return grid


def ensure_priority_grid_exists(n: int, output_dir: str = 'priority_grids') -> np.ndarray:
    """
    Ensure a priority grid exists for size n, computing it if necessary.
    
    Args:
        n (int): Grid size
        output_dir (str): Directory to store/load priority grids
        
    Returns:
        np.ndarray: The priority grid
    """
    try:
        return load_priority_grid(n, output_dir)
    except FileNotFoundError:
        print(f"Priority grid for n={n} not found. Computing...")
        grid = priority_grid_advanced(n)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'priority_grid_{n}.npy')
        np.save(filename, grid)
        print(f"Saved priority_grid_{n}.npy")
        return grid


def find_missing_data(data: Dict[str, Iterable]) -> Dict[str, Iterable]:
    """
    Identify entries with missing data labeled as 'NO DATA' in intervals.

    Args:
        data (Dict[str, Iterable]): Dictionary mapping keys to iterables of (start, end) intervals.

    Returns:
        Dict[str, Iterable]: Dictionary mapping keys to iterables of tuples containing 
                            the index of the interval and the start value where 'NO DATA' occurs.
    """
    missing = {}
    for key, intervals in data.items():
        missing_entries = []
        for i, (start, end) in enumerate(intervals):
            if end == 'NO DATA':
                missing_entries.append((i, start))
        if missing_entries:
            missing[key] = missing_entries
    return missing
