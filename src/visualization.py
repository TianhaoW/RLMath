"""
Visualization utilities for grid-based no-three-collinear problems.
Extracted from luoning's MCTS along priority notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Optional

from .geometry import Point


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


def plot_priority_heatmap(priority_grid: np.ndarray, title: str = "Priority Heatmap") -> None:
    """
    Plot a heatmap of priority values for a grid.

    Args:
        priority_grid (np.ndarray): 2D array of priority values.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(priority_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Priority Value')
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # Add grid lines
    n = priority_grid.shape[0]
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.grid(True, color='white', linewidth=0.5, alpha=0.3)
    
    # Add text annotations for each cell
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f'{priority_grid[i, j]:.1f}', 
                    ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.show()
