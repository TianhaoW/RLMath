import itertools
import numpy as np
print(np.__version__)
np.random.seed(0)
from tqdm import trange
from numba import njit
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
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

import psutil
import os
import csv
import pandas as pd

def count_idle_cpus(threshold: float = 10.0) -> int:
    """
    Count CPU cores with usage below the threshold.

    Args:
        threshold (float): Utilization percentage below which a core is considered idle.

    Returns:
        int: Number of idle CPU cores.
    """
    usage: List[float] = psutil.cpu_percent(percpu=True)
    # return sum(1 for u in usage if u < threshold)
    return 1

CPU_CORES = 4

def binomial(n, k):
    if hasattr(math, "comb"):
        return math.comb(n, k)
    # Fallback for Python <3.8
    if 0 <= k <= n:
        num = 1
        denom = 1
        for i in range(1, k+1):
            num *= n - (i - 1)
            denom *= i
        return num // denom
    return 0

class QQ:
    def __init__(self, numerator, denominator=1):
        if denominator == 0:
            raise ZeroDivisionError("Denominator cannot be zero.")
        if not isinstance(numerator, int):
            numerator = Integer(numerator)
        if not isinstance(denominator, int):
            denominator = Integer(denominator)
        g = igcd(numerator, denominator)
        self.num = Integer(numerator // g)
        self.den = Integer(denominator // g)
        if self.den < 0:
            self.num = -self.num
            self.den = -self.den

    def __add__(self, other):
        if not isinstance(other, QQ):
            other = QQ(other)
        num = self.num * other.den + other.num * self.den
        den = self.den * other.den
        return QQ(num, den)

    def __sub__(self, other):
        if not isinstance(other, QQ):
            other = QQ(other)
        num = self.num * other.den - other.num * self.den
        den = self.den * other.den
        return QQ(num, den)

    def __mul__(self, other):
        if not isinstance(other, QQ):
            other = QQ(other)
        return QQ(self.num * other.num, self.den * other.den)

    def __truediv__(self, other):
        if not isinstance(other, QQ):
            other = QQ(other)
        if other.num == 0:
            raise ZeroDivisionError("Division by zero.")
        return QQ(self.num * other.den, self.den * other.num)

    def __neg__(self):
        return QQ(-self.num, self.den)

    def __eq__(self, other):
        if isinstance(other, str):
            return False
        if not isinstance(other, QQ):
            other = QQ(other)
        return self.num == other.num and self.den == other.den

    def __float__(self):
        return float(self.num) / float(self.den)

    def __repr__(self):
        return f"{self.num}/{self.den}" if self.den != 1 else f"{self.num}"
    
    def __hash__(self):
        return hash((self.num, self.den))

    def to_sympy(self):
        return Rational(self.num, self.den)

class Point(NamedTuple):
    """An integer point in 2D space."""
    x: int
    y: int

class Line:
    """
    Represents a line defined by a rational slope and a point on the line.

    Attributes:
        slope (Union[QQ, str]): Rational slope of the line, or 'inf' for vertical lines.
        point (Point): An arbitrary point on the line.
    """

    def __init__(self, slope: Union[QQ, str], point: Point):
        """
        Initialize a line with a given slope and a point on the line.

        Args:
            slope (Union[QQ, str]): Rational slope of the line, or 'inf' for vertical lines.
            point (Point): A point on the line.
        """
        self.slope = slope
        self.point = point

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> 'Line':
        """
        Construct a line from two points.

        The slope is computed from the two points. The stored point is p1 without any minimization.

        Args:
            p1 (Point): First point.
            p2 (Point): Second point.

        Returns:
            Line: Line through p1 and p2 with p1 stored as the point.
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        if dx == 0:
            slope = 'inf'
        else:
            slope = QQ(dy) / QQ(dx)
        return cls(slope, p1)

    @classmethod
    def from_point_slope_of_line(cls, p: Point, slope: Union[QQ, str]) -> 'Line':
        """
        Construct a line from a point and a slope.

        Args:
            p (Point): A point on the line.
            slope (Union[QQ, str]): Rational slope of the line, or 'inf' for vertical lines.

        Returns:
            Line: Line defined by the point and slope.
        """
        return cls(slope, p)


def slope_of_line(p1: Point, p2: Point) -> Union[QQ, str]:
    """
    Calculate the slope of the line segment connecting two points.

    Args:
        p1 (Point): The first point as a named tuple with integer coordinates (x, y).
        p2 (Point): The second point as a named tuple with integer coordinates (x, y).

    Returns:
        Union[QQ, str]: The slope as a rational number (QQ) if defined, otherwise the string 'inf' if the line is vertical.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 'inf'
    return QQ(dy) / QQ(dx)

def are_collinear(p1: Point, p2: Point, p3: Point) -> bool:
    """
    Determine if three points are collinear.

    Args:
        p1 (Point): The first point as a named tuple with integer coordinates (x, y).
        p2 (Point): The second point as a named tuple with integer coordinates (x, y).
        p3 (Point): The third point as a named tuple with integer coordinates (x, y).

    Returns:
        bool: True if the three points are collinear, False otherwise.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

def plot_no_three_in_line(points: Iterable[Point], n: Optional[int] = None, title: str = "No-3-in-line Set") -> None:
    """
    Plot a set of points on an n x n grid, illustrating a no-3-in-line configuration.

    Args:
        points (Iterable[Point]): Iterable of points to plot.
        n (Optional[int]): Size of the grid. If None, computed from the points.
        title (str): Title of the plot.

    Returns:
        None
    """
    points = list(points)
    if not points:
        print("No points to plot.")
        return

    xs, ys = zip(*points)
    if n is None:
        n = max(max(xs), max(ys)) + 1

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=100, c='blue', edgecolors='black')
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1, n)
    plt.ylim(-1, n)
    plt.show()

def find_missing_data(data: Dict[str, Iterable[Tuple[Union[int, str], Union[int, str]]]]) -> Dict[str, Iterable[Tuple[int, Union[int, str]]]]:
    """
    Identify entries with missing data labeled as 'NO DATA' in intervals.

    Args:
        data (Dict[str, Iterable[Tuple[Union[int, str], Union[int, str]]]]): Dictionary mapping keys to iterables of (start, end) intervals.

    Returns:
        Dict[str, Iterable[Tuple[int, Union[int, str]]]]: Dictionary mapping keys to iterables of tuples containing the index of the interval and the start value where 'NO DATA' occurs.
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

def points_on_line_pp(p1: Point, p2: Point, grid_size: int) -> Set[Point]:
    """
    Generate points with integer coordinates on the line segment between two points,
    assuming the segment lies on a line with rational slope and intercept.

    Args:
        p1 (Point): The first point as a named tuple with integer coordinates (x, y).
        p2 (Point): The second point as a named tuple with integer coordinates (x, y).

    Returns:
        Set[Point]: Set of points on the line segment from p1 to p2.
    """
    s = slope_of_line(p1, p2)
    if s == 'inf':
        x = p1.x
        return {Point(x, y) for y in range(grid_size) if 0 <= x < grid_size}
    a = s
    b = QQ(p1.y) - a * QQ(p1.x)
    return {Point(x, int(y)) for x in range(grid_size)
            if (y := a * QQ(x) + b).denominator() == 1 and 0 <= y < grid_size}

def points_on_line_l(line: Line, grid_size: int) -> Set[Point]:
    """
    Generate points with integer coordinates on the line defined by the given line object.

    Args:
        line (Line): The line object representing the line.
        grid_size (int): The size of the grid.

    Returns:
        Set[Point]: Set of points on the line within the grid size.
    """
    a = line.slope
    p = line.point
    if a == 'inf':
        x = p.x
        return {Point(x, y) for y in range(grid_size) if 0 <= x < grid_size}
    b = QQ(p.y) - a * QQ(p.x)
    return {Point(x, int(y)) for x in range(grid_size)
            if (y := a * QQ(x) + b).denominator() == 1 and 0 <= y < grid_size}

def __eq__(self, other):
    if isinstance(other, str):
        return False
    if not isinstance(other, QQ):
        other = QQ(other)
    return self.num == other.num and self.den == other.den

def count_points_on_line(p: Point, slope: Union[QQ, str], grid_size: int) -> int:
    """
    Count the number of integer points (excluding point p) on line defined by an intersection point and slope.

    Args:
        p (Point): The given point the line passes through.
        slope (Union[QQ, str): The slope of the line (non-negative), either as a rational number (QQ) or 'inf' for vertical lines.
        grid_size (int): The size of the grid.

    Returns:
        int: The number of integer points on the line.
    """
    if min(p.x, p.y) < 0 or max(p.x, p.y) >= grid_size:
        return 0
    if slope == 'inf' or slope == 0:
        return grid_size - 1

    dy = abs(slope.num)
    dx = abs(slope.den)
    U = math.floor((grid_size - p.x - 1) / dx)
    R = math.floor((grid_size - p.y - 1) / dy)

    D = math.floor(p.x / dx)
    L = math.floor(p.y / dy)
    N_positive = min(U, R) + min(D, L) 
    N_negative = min(U, L) + min(D, R)

    if (N_positive if slope.num > 0 else N_negative) < 0:
        print(f"point: {p}, slope: {slope}")
        print(f"U: {U}, R: {R}, D: {D}, L: {L}, N_positive: {N_positive}, N_negative: {N_negative}")
    return N_positive if slope.num > 0 else N_negative

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

def get_possible_slopes(grid_size: int, idle_cores: int = 0) -> Set[Union[QQ, str]]:
    """
    Generate all possible slopes for lines on a grid of given size.

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


def point_collinear_count(p1: Point, grid_size: int) -> int:
    """
    Count the number of collinear triples on that line including that point (not including horizontal and vertical line).

    Args:
        p1 (Point): The point to check.
        grid_size (int): The size of the grid.

    Returns:
        int: Sum over slopes of binomial(count, 2) for points collinear with p1.
    """
    idle = count_idle_cpus()
    slopes = get_possible_slopes(grid_size, idle_cores=idle)
    counts = sum([
        binomial(count_points_on_line(p1, slope, grid_size), 2)
        for slope in slopes if slope != 0 and slope != 'inf'
    ])
    return counts

# NOISE = random.uniform(-0.1, 0.1) 
NOISE =  0

def collinear_count_priority(n):
    def priority(point):
        return -point_collinear_count(point, n) + NOISE
    return priority

def priority_grid(n):
    """
    Return a 2D numpy array of priority values for each point in an n x n grid.

    Args:
        n (int): The grid size.

    Returns:
        np.ndarray: 2D array of priority values.
    """
    priority_fn = collinear_count_priority(n)
    arr = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            arr[x, y] = priority_fn(Point(x, y))
    return arr

# Example usage:
# grid = priority_grid(3)
# print(grid)

def heatmap(grid_values: np.ndarray, title: str = "") -> None:
    """
    Plot a heatmap from a 2D numpy array of values and save it to a file named heatmap{n}.png.
    
    Args:
        grid_values (np.ndarray): 2D array of values to plot.
        title (str): Title of the heatmap plot.
    """
    import matplotlib.pyplot as plt

    n = grid_values.shape[0]
    plt.figure(figsize=(6,6))
    plt.imshow(grid_values, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Priority')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.show()

def compute_and_save_priority_grids(priority_grid_fn, size_list=None, output_dir='priority_grids'):
    """
    Compute priority grids for each n in size_list (if not already saved),
    and save them as .npy files in the output_dir directory.

    Parameters:
    - priority_grid_fn: A function to generate the priority grid, which takes n as input.
    - size_list: List of n values to compute.
    - output_dir: Directory where .npy files will be saved.
    """
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


def load_priority_grid(n, input_dir='priority_grids'):
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

@njit(cache=True)
def supnorm_priority(x: int, y: int) -> float:
    """
    Compute the negative sup-norm priority for a single point.
    Equivalent to: -abs(max(x, y))
    """
    m = x if x > y else y
    return abs(m)

@njit(cache=True)
def supnorm_priority_array(n: int) -> np.ndarray:
    """
    Generate an n-by-n array of priorities using the supnorm_priority function.
    """
    arr = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            arr[i, j] = supnorm_priority(i, j)
    return arr

@njit(cache=True)
def value_fn_nb(x):
    # return x
    # return np.exp(x)
    return x

@njit(cache=True)
def get_value_nb(state, pts_upper_bound, value_f=value_fn_nb):
    total = np.sum(state)
    n = pts_upper_bound/2
    # return value_f(total) / value_f(pts_upper_bound)
    return (total - 1.5 * n) / (0.5 * n)

# JIT-compiled function to check if three points are collinear
@njit(cache=True)
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


# JIT-compiled function to perform a full random rollout until terminal
@njit(cache=True, nogil=True)
def simulate_nb(state, row_count, column_count, pts_upper_bound):
    """
    Perform random rollout until no valid moves remain.
    Return normalized value using a custom value function.
    Uses get_valid_moves_subset_nb for incremental validity updates.
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

# This is not gym-compatible, but serves as a base class for MCTS-like algorithms
class N3il:
    def __init__(self, grid_size, args, priority_grid=None):
        self.row_count, self.column_count = grid_size
        self.pts_upper_bound = np.min(grid_size) * 2
        self.action_size = self.row_count * self.column_count
        self.args = args
        self.priority_grid = priority_grid  # Store priority grid
        
        # Create session name with timestamp and grid size
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{timestamp}_{self.row_count}by{self.column_count}"

    def state_to_key(self, state):
        """Convert state to a hashable key for the node registry."""
        return tuple(state.flatten())

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), np.uint8)

    def get_next_state(self, state, action):
        row = action // self.column_count
        col = action % self.column_count
        state[row, col] = 1
        return state

    def get_valid_moves(self, state):
        # Get all valid moves
        valid_moves = get_valid_moves_nb(state, self.row_count, self.column_count)
        # Only keep moves with the highest priority
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        else:
            return valid_moves

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken):
        # Get all valid moves
        valid_moves = get_valid_moves_subset_nb(parent_state, parent_valid_moves, action_taken, self.row_count, self.column_count)
        # Only keep moves with the highest priority
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        else:
            return valid_moves

    def check_collinear(self, state, action=None):
        if action is not None:
            temp_state = state.copy()
            row = action // self.column_count
            col = action % self.column_count
            temp_state[row, col] = 1
        else:
            temp_state = state

        # Call numba-accelerated function
        return check_collinear_nb(temp_state, self.row_count, self.column_count)

    def get_value_and_terminated(self, state, valid_moves):
        """
        Return the normalized value and terminal status of the current state.
        Delegates value calculation to get_value_nb().
        """
        if np.sum(valid_moves) > 0:
            return 0.0, False

        value = get_value_nb(state, self.pts_upper_bound)
        return value, True
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state

    def display_state(self, state, action_prob=None):
        """
        Display the current grid configuration using matplotlib.
        Points are drawn where the state equals 1.
        The origin (0, 0) is located at the bottom-left.
        If action_prob is provided (1D array), it is reshaped and overlaid as a heatmap.
        Marker sizes, font sizes, and grid line widths auto-adjust to the grid dimensions.
        """
        rows, cols = self.row_count, self.column_count

        # Create date-time folder once per instance
        if not hasattr(self, '_display_folder'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{timestamp}_{rows}by{cols}"
            base_dir = self.args.get('figure_dir', 'figures')
            self._display_folder = os.path.join(base_dir, folder_name)
            os.makedirs(self._display_folder, exist_ok=True)

        # keep a fixed figure size
        plt.figure(figsize=(12, 10))
        ax = plt.gca()

        # dynamically compute sizes (fixed fig)
        marker_size     = 50000.0 / (rows * cols)      # scatter dot size
        font_size       = 200.0    / max(rows, cols)   # annotation font size
        grid_line_width = 2.0      / max(rows, cols)   # grid line width
        cb_shrink       = 0.8                        # colorbar shrink
        
        # Calculate proper DPI based on grid size for clarity while maintaining speed
        grid_area = rows * cols
        if grid_area <= 100:      # 10x10 or smaller
            dpi = 150
        elif grid_area <= 400:    # 20x20 or smaller
            dpi = 200
        elif grid_area <= 900:    # 30x30 or smaller
            dpi = 250
        else:                     # larger grids
            dpi = min(300, 150 + (grid_area - 900) // 50)  # gradually increase, cap at 300
        
        # Calculate title font size - bigger for larger grids
        title_font_size = max(10, min(20, 8 + max(rows, cols) * 0.3))

        if action_prob is not None:
            assert action_prob.shape[0] == rows * cols, \
                f"Expected length {rows * cols}, got {len(action_prob)}"
            action_prob_2d = action_prob.reshape((rows, cols))
            flipped = np.flipud(action_prob_2d)

            im = ax.imshow(
                flipped,
                cmap='Reds',
                alpha=0.6,
                extent=[-0.5, cols - 0.5, -0.5, rows - 0.5],
                origin='lower',
                vmin=0, vmax=action_prob.max() if action_prob.max() > 0 else 1e-5
            )
            plt.colorbar(im, label="Action Probability", shrink=cb_shrink)

            max_val = action_prob_2d.max()
            max_positions = np.argwhere(action_prob_2d == max_val)

            # annotate each cell
            for i in range(rows):
                for j in range(cols):
                    val = action_prob_2d[i, j]
                    y_disp = rows - 1 - i
                    is_max = any((i, j) == tuple(mp) for mp in max_positions)
                    color = 'gold' if is_max else ('white' if val > 0.5 * max_val else 'black')
                    weight = 'bold' if is_max else 'normal'
                    ax.text(
                        j, y_disp, f"{val:.2f}",
                        ha='center', va='center',
                        color=color, weight=weight,
                        fontsize=font_size
                    )

        # plot placed points
        y_idx, x_idx = np.nonzero(state)
        y_disp = rows - 1 - y_idx
        ax.scatter(x_idx, y_disp, s=marker_size, c='blue', linewidths=0.5)

        # draw grid lines
        ax.set_xticks(np.arange(-0.5, cols, 1))
        ax.set_yticks(np.arange(-0.5, rows, 1))
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=grid_line_width)
        ax.set_xticks([])  # hide tick labels
        ax.set_yticks([])

        # set equal aspect and limits
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')

        # Create comprehensive title with all args info and points count
        num_points = np.sum(state)
        base_title = "No-Three-In-Line Grid with Action Probabilities" if action_prob is not None else "No-Three-In-Line Grid"
        
        # Extract all key info from args
        algorithm = self.args.get('algorithm', 'Unknown')
        num_searches = self.args.get('num_searches', 'N/A')
        C = self.args.get('C', 'N/A')
        topN = self.args.get('TopN', 'N/A')
        priority = self.args.get('simulate_with_priority', 'N/A')
        workers = self.args.get('num_workers', 'N/A')
        
        title = f"{base_title}\nGrid: {rows}×{cols}, Points: {num_points}, Algorithm: {algorithm}, Searches: {num_searches}\nC: {C}, TopN: {topN}, Priority: {priority}, Workers: {workers}"
        plt.title(title, fontsize=title_font_size, pad=20)
        
        plt.tight_layout()
        
        # Save the plot with timestamp, grid size, and number of points for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"no_three_in_line_{rows}x{cols}_pts{num_points}_{algorithm}_{timestamp}.png"
        
        # Full path for the file
        full_path = os.path.join(self._display_folder, filename)
        
        try:
            plt.savefig(full_path, format='png', dpi=dpi, bbox_inches='tight')
            print(f"Plot saved as: {full_path} (DPI: {dpi})")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()  # Close the figure to free memory

    def record_to_table(self, terminal_num_points, start_time, end_time, time_used):
        """
        Record experiment results to a CSV table in the table_dir directory.
        Creates the CSV file if it doesn't exist, otherwise appends data.
        """
        import csv
        import pandas as pd
        
        # Get table directory and create if it doesn't exist
        table_dir = self.args.get('table_dir', 'results')
        os.makedirs(table_dir, exist_ok=True)
        
        # CSV file path
        csv_file = os.path.join(table_dir, 'experiment_results.csv')
        
        # Prepare data row
        data_row = {}
        
        # Add all args as columns
        for key, value in self.args.items():
            data_row[key] = value
        
        # Add additional required columns
        data_row['terminal_num_points'] = terminal_num_points
        data_row['session_name'] = self.session_name
        data_row['start_time'] = start_time
        data_row['end_time'] = end_time
        data_row['time_used'] = time_used
        
        # Check if CSV file exists
        file_exists = os.path.exists(csv_file)
        
        try:
            if file_exists:
                # Read existing CSV to get all possible columns
                existing_df = pd.read_csv(csv_file)
                all_columns = list(existing_df.columns)
                
                # Check if there are new columns from current data_row
                new_columns = [key for key in data_row.keys() if key not in all_columns]
                
                if new_columns:
                    # Add new columns to existing DataFrame with null values
                    for col in new_columns:
                        existing_df[col] = None
                        all_columns.append(col)
                    
                    # Save the updated DataFrame with new columns
                    existing_df.to_csv(csv_file, index=False)
                
                # Create new row with all columns (fill missing with None)
                new_row = {}
                for col in all_columns:
                    new_row[col] = data_row.get(col, None)
                
                # Append to existing CSV
                new_df = pd.DataFrame([new_row])
                new_df.to_csv(csv_file, mode='a', header=False, index=False)
                
            else:
                # Create new CSV file
                df = pd.DataFrame([data_row])
                df.to_csv(csv_file, index=False)
            
            print(f"Results recorded to: {csv_file}")
            
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            # Fallback: try simple CSV writing
            try:
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data_row.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(data_row)
                print(f"Results recorded to: {csv_file} (fallback method)")
            except Exception as e2:
                print(f"Error with fallback CSV writing: {e2}")
