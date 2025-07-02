"""
Priority-based utilities for No-3-In-Line
Extracted from nothreeinline-Spring-25 notebooks
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import Callable, List, Tuple, Dict, Any
import random
from itertools import combinations


def are_collinear(p1, p2, p3):
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


def slope(p1, p2):
    """Calculate slope between two points"""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return float('inf')
    return dy / dx


def square_corner_priority(n, sharpness=1, radius_scale=1.0):
    """
    Square corner priority function
    Gives higher priority to points near corners
    
    Args:
        n: Grid size
        sharpness: Controls how sharp the priority gradient is
        radius_scale: Scales the effective radius
        
    Returns:
        Priority function
    """
    def priority_func(point):
        x, y = point
        # Distance to nearest corner
        corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
        min_dist = min(np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in corners)
        
        # Convert to priority (closer to corner = higher priority)
        max_dist = np.sqrt(2) * (n - 1) / 2  # Maximum possible distance to corner
        normalized_dist = min_dist / max_dist
        priority = (1 - normalized_dist) ** sharpness
        
        return priority
    
    return priority_func


def center_priority(n, sharpness=2):
    """
    Center priority function
    Gives higher priority to points near the center
    
    Args:
        n: Grid size
        sharpness: Controls how sharp the priority gradient is
        
    Returns:
        Priority function
    """
    center = (n - 1) / 2
    max_dist = np.sqrt(2) * center
    
    def priority_func(point):
        x, y = point
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        normalized_dist = dist / max_dist
        priority = (1 - normalized_dist) ** sharpness
        return priority
    
    return priority_func


def edge_priority(n, edge_bonus=2.0):
    """
    Edge priority function
    Gives higher priority to points on the edges
    
    Args:
        n: Grid size
        edge_bonus: Bonus multiplier for edge points
        
    Returns:
        Priority function
    """
    def priority_func(point):
        x, y = point
        
        # Check if on edge
        on_edge = (x == 0 or x == n-1 or y == 0 or y == n-1)
        
        if on_edge:
            return edge_bonus
        else:
            # Distance to nearest edge
            dist_to_edge = min(x, y, n-1-x, n-1-y)
            max_dist = (n - 1) // 2
            normalized_dist = dist_to_edge / max_dist
            return normalized_dist
    
    return priority_func


def sup_norm_priority(n, center_boost=1.5):
    """
    Supremum norm priority function
    Uses L-infinity distance metric
    
    Args:
        n: Grid size
        center_boost: Boost factor for center region
        
    Returns:
        Priority function
    """
    center = (n - 1) / 2
    
    def priority_func(point):
        x, y = point
        # L-infinity distance from center
        sup_dist = max(abs(x - center), abs(y - center))
        max_sup_dist = center
        
        normalized_dist = sup_dist / max_sup_dist
        priority = (1 - normalized_dist) * center_boost
        
        return priority
    
    return priority_func


def greedy_no_three_inline(n, priority_func, noise_factor=0.0):
    """
    Greedy algorithm for no-3-in-line using priority function
    
    Args:
        n: Grid size
        priority_func: Function that assigns priority to each point
        noise_factor: Amount of noise to add to priorities
        
    Returns:
        List of selected points
    """
    all_points = [(x, y) for x in range(n) for y in range(n)]
    
    # Add noise if specified
    if noise_factor > 0:
        noise = {p: random.uniform(-noise_factor, noise_factor) for p in all_points}
    else:
        noise = {p: 0 for p in all_points}
    
    priorities = {p: priority_func(p) + noise[p] for p in all_points}
    active_points = set(all_points)
    selected = set()
    
    while active_points:
        # Select point with highest priority
        p = max(active_points, key=lambda pt: priorities[pt])
        selected.add(p)
        active_points.remove(p)
        
        # Remove points that would form collinear triples
        to_remove = set()
        for q in active_points:
            for s in selected:
                if s != p and are_collinear(p, s, q):
                    to_remove.add(q)
                    break
        
        active_points -= to_remove
    
    return sorted(selected)


def plot_no_three_in_line(points, n=None, title="No-3-in-line Set", save_path=None):
    """
    Plot a no-3-in-line configuration
    
    Args:
        points: List of (x, y) points
        n: Grid size (auto-detected if None)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not points:
        print("No points to plot.")
        return
    
    xs, ys = zip(*points)
    if n is None:
        n = max(max(xs), max(ys)) + 1
    
    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, s=100, c='blue', edgecolors='black', zorder=5)
    
    # Add grid
    for i in range(n):
        plt.axhline(y=i, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{title} ({len(points)} points)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-0.5, n-0.5)
    plt.ylim(-0.5, n-0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_priority_grid(priority_func, n, filename):
    """
    Save priority values for all grid points
    
    Args:
        priority_func: Priority function
        n: Grid size
        filename: File to save to
    """
    priority_grid = np.zeros((n, n))
    
    for x in range(n):
        for y in range(n):
            priority_grid[x, y] = priority_func((x, y))
    
    with open(filename, 'wb') as f:
        pickle.dump(priority_grid, f)
    
    return priority_grid


def load_priority_grid(filename):
    """Load priority grid from file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def visualize_priority_heatmap(priority_func, n, title="Priority Heatmap", save_path=None):
    """
    Visualize priority function as a heatmap
    
    Args:
        priority_func: Priority function
        n: Grid size
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    priority_grid = np.zeros((n, n))
    
    for x in range(n):
        for y in range(n):
            priority_grid[x, y] = priority_func((x, y))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(priority_grid, cmap='viridis', origin='lower')
    plt.colorbar(label='Priority')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return priority_grid


def compare_priority_functions(priority_funcs, n, num_trials=10):
    """
    Compare different priority functions
    
    Args:
        priority_funcs: Dictionary of {name: priority_function}
        n: Grid size
        num_trials: Number of trials per function
        
    Returns:
        Dictionary with results
    """
    results = {}
    
    for name, priority_func in priority_funcs.items():
        scores = []
        configurations = []
        
        for trial in range(num_trials):
            points = greedy_no_three_inline(n, priority_func, noise_factor=0.1)
            scores.append(len(points))
            configurations.append(points)
        
        results[name] = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'best_score': max(scores),
            'best_config': configurations[np.argmax(scores)]
        }
    
    return results


def print_comparison_results(results):
    """Print comparison results in a nice format"""
    print("Priority Function Comparison Results:")
    print("=" * 50)
    
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Mean score: {data['mean_score']:.2f} ± {data['std_score']:.2f}")
        print(f"  Best score: {data['best_score']}")
        print()


# Predefined priority function configurations
PRIORITY_FUNCTIONS = {
    'square_corner': lambda n: square_corner_priority(n, sharpness=1),
    'center': lambda n: center_priority(n, sharpness=2),
    'edge': lambda n: edge_priority(n, edge_bonus=2.0),
    'sup_norm': lambda n: sup_norm_priority(n, center_boost=1.5),
}
