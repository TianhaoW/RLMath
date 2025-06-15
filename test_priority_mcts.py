"""
Test script for the extracted priority-based MCTS functionality.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.geometry import Point, QQ, are_collinear
from src.priority import priority_grid, collinear_count_priority, point_collinear_count
from src.visualization import plot_no_three_in_line, plot_priority_heatmap
from src.algos.mcts_priority import MCTSPriorityAgent


def test_geometry():
    """Test basic geometry functions."""
    print("Testing geometry functions...")
    
    # Test QQ class
    q1 = QQ(1, 2)
    q2 = QQ(3, 4)
    print(f"QQ(1,2) + QQ(3,4) = {q1 + q2}")
    print(f"QQ(1,2) * QQ(3,4) = {q1 * q2}")
    
    # Test collinearity
    p1, p2, p3 = Point(0, 0), Point(1, 1), Point(2, 2)
    p4 = Point(1, 2)
    print(f"Points {p1}, {p2}, {p3} are collinear: {are_collinear(p1, p2, p3)}")
    print(f"Points {p1}, {p2}, {p4} are collinear: {are_collinear(p1, p2, p4)}")


def test_priority():
    """Test priority calculation functions."""
    print("\nTesting priority functions...")
    
    # Test for small grid
    grid_size = 3
    
    # Calculate priority for center point
    center_point = Point(1, 1)
    collinear_count = point_collinear_count(center_point, grid_size)
    print(f"Collinear count for {center_point} in {grid_size}x{grid_size} grid: {collinear_count}")
    
    # Generate priority grid
    priorities = priority_grid(grid_size)
    print(f"Priority grid for {grid_size}x{grid_size}:")
    print(priorities)


def test_visualization():
    """Test visualization functions."""
    print("\nTesting visualization...")
    
    # Create some sample points
    points = [Point(0, 0), Point(1, 2), Point(2, 1)]
    plot_no_three_in_line(points, n=3, title="Sample No-3-in-line Configuration")
    
    # Create and plot priority heatmap
    priorities = priority_grid(4)
    plot_priority_heatmap(priorities, title="4x4 Priority Heatmap")


def test_mcts():
    """Test MCTS priority agent."""
    print("\nTesting MCTS Priority Agent...")
    
    # Create agent for small grid
    grid_size = 4
    agent = MCTSPriorityAgent(grid_size=grid_size, max_iterations=100)
    
    # Play a game
    points, score = agent.play_game(verbose=True)
    print(f"Game completed with {score} points placed")
    print(f"Final points: {points}")
    
    # Visualize result
    plot_no_three_in_line(points, n=grid_size, title=f"MCTS Result: {score} points")


def main():
    """Run all tests."""
    print("Testing extracted functionality from luoning's notebook...")
    
    test_geometry()
    test_priority()
    
    # Note: Visualization tests will show plots
    # Uncomment to run visualization tests
    # test_visualization()
    # test_mcts()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
