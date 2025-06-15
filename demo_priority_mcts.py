"""
Demonstration of the extracted priority-based MCTS functionality.
This integrates luoning's notebook functions into the src framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

from src.geometry import Point, QQ, are_collinear, slope_of_line, Line
from src.priority import priority_grid, collinear_count_priority, point_collinear_count, get_possible_slopes
from src.visualization import plot_no_three_in_line, plot_priority_heatmap
from src.algos.mcts_priority import MCTSPriorityAgent


def demo_geometry_functions():
    """Demonstrate geometry functions extracted from the notebook."""
    print("=== Geometry Functions Demo ===")
    
    # QQ (Rational number) arithmetic
    print("\n1. Rational Number (QQ) Arithmetic:")
    q1 = QQ(1, 3)
    q2 = QQ(2, 5)
    print(f"   {q1} + {q2} = {q1 + q2}")
    print(f"   {q1} * {q2} = {q1 * q2}")
    print(f"   {q1} / {q2} = {q1 / q2}")
    
    # Point and Line operations
    print("\n2. Point and Line Operations:")
    p1, p2, p3 = Point(0, 0), Point(2, 1), Point(4, 2)
    print(f"   Points: {p1}, {p2}, {p3}")
    
    slope = slope_of_line(p1, p2)
    print(f"   Slope from {p1} to {p2}: {slope}")
    
    line = Line.from_points(p1, p2)
    print(f"   Line through {p1} and {p2}: slope={line.slope}, point={line.point}")
    
    # Collinearity check
    print(f"   Are {p1}, {p2}, {p3} collinear? {are_collinear(p1, p2, p3)}")
    
    p4 = Point(1, 2)
    print(f"   Are {p1}, {p2}, {p4} collinear? {are_collinear(p1, p2, p4)}")


def demo_priority_functions():
    """Demonstrate priority calculation functions."""
    print("\n=== Priority Functions Demo ===")
    
    grid_size = 5
    print(f"\nAnalyzing {grid_size}x{grid_size} grid:")
    
    # Calculate priority for specific points
    points_to_analyze = [Point(0, 0), Point(2, 2), Point(1, 3), Point(4, 1)]
    
    print("\nCollinear counts for specific points:")
    for point in points_to_analyze:
        count = point_collinear_count(point, grid_size)
        print(f"   {point}: {count} potential collinear triples")
    
    # Generate full priority grid
    priorities = priority_grid(grid_size)
    print(f"\nFull priority grid for {grid_size}x{grid_size}:")
    print(priorities)
    
    # Find best and worst positions
    best_pos = np.unravel_index(np.argmax(priorities), priorities.shape)
    worst_pos = np.unravel_index(np.argmin(priorities), priorities.shape)
    print(f"\nBest position (highest priority): {Point(best_pos[1], best_pos[0])} with priority {priorities[best_pos]}")
    print(f"Worst position (lowest priority): {Point(worst_pos[1], worst_pos[0])} with priority {priorities[worst_pos]}")
    
    return priorities


def demo_mcts_agent():
    """Demonstrate the MCTS agent with priority-based selection."""
    print("\n=== MCTS Priority Agent Demo ===")
    
    grid_size = 6
    print(f"\nRunning MCTS on {grid_size}x{grid_size} grid...")
    
    # Create agent with different configurations
    configurations = [
        {"max_iterations": 50, "c_param": 1.0, "name": "Fast"},
        {"max_iterations": 200, "c_param": 1.4, "name": "Balanced"},
        {"max_iterations": 500, "c_param": 2.0, "name": "Thorough"}
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{config['name']} MCTS (iterations={config['max_iterations']}, c_param={config['c_param']}):")
        
        agent = MCTSPriorityAgent(
            grid_size=grid_size, 
            max_iterations=config['max_iterations'],
            c_param=config['c_param']
        )
        
        points, score = agent.play_game(verbose=False)
        results.append((config['name'], points, score))
        
        print(f"   Result: {score} points placed")
        print(f"   Points: {points}")
    
    # Find best result
    best_result = max(results, key=lambda x: x[2])
    print(f"\nBest result: {best_result[0]} configuration with {best_result[2]} points")
    
    return results


def demo_slopes_analysis():
    """Demonstrate slope analysis for different grid sizes."""
    print("\n=== Slope Analysis Demo ===")
    
    for grid_size in [3, 4, 5]:
        slopes = get_possible_slopes(grid_size)
        print(f"\n{grid_size}x{grid_size} grid:")
        print(f"   Total unique slopes: {len(slopes)}")
        
        # Categorize slopes
        rational_slopes = [s for s in slopes if isinstance(s, QQ)]
        special_slopes = [s for s in slopes if isinstance(s, str)]
        
        print(f"   Rational slopes: {len(rational_slopes)}")
        print(f"   Special slopes: {special_slopes}")
        
        # Show a few example rational slopes
        if rational_slopes:
            print(f"   Example rational slopes: {sorted(rational_slopes, key=lambda x: float(x))[:5]}")


def create_comparison_visualization():
    """Create visualizations comparing different approaches."""
    print("\n=== Creating Comparison Visualizations ===")
    
    grid_size = 5
    
    # Generate priority heatmap
    priorities = priority_grid(grid_size)
    
    # Run MCTS to get a solution
    agent = MCTSPriorityAgent(grid_size=grid_size, max_iterations=200)
    points, score = agent.play_game()
    
    print(f"Visualization created for {grid_size}x{grid_size} grid")
    print(f"MCTS found solution with {score} points")
    print("Note: Uncomment the plot functions in the code to see visualizations")
    
    # Uncomment these lines to show plots:
    # plot_priority_heatmap(priorities, f"{grid_size}x{grid_size} Priority Heatmap")
    # plot_no_three_in_line(points, grid_size, f"MCTS Solution: {score} Points")
    
    return priorities, points


def integration_with_existing_env():
    """Show how to integrate with existing collinear environment."""
    print("\n=== Integration with Existing Environment ===")
    
    try:
        from src.envs.colinear import NoThreeCollinearEnv
        
        # Create environment
        env = NoThreeCollinearEnv(m=5, n=5)
        
        # Use priority function to suggest moves
        priority_fn = collinear_count_priority(5)
        
        print("Environment created successfully")
        print("Priority function can be used to guide RL agents")
        
        # Example: Get valid actions and their priorities
        obs, info = env.reset()
        valid_actions = []
        for action in range(env.action_space.n):
            point = env.decode_action(action)
            if not env.is_selected(point):
                valid_actions.append((action, point, priority_fn(point)))
        
        # Sort by priority
        valid_actions.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop 5 recommended actions by priority:")
        for i, (action, point, priority) in enumerate(valid_actions[:5]):
            print(f"   {i+1}. Action {action} -> {point} (priority: {priority:.2f})")
            
    except ImportError:
        print("NoThreeCollinearEnv not available for integration demo")


def main():
    """Run all demonstrations."""
    print("Demonstrating Extracted Priority-Based MCTS Functionality")
    print("=" * 60)
    
    demo_geometry_functions()
    priorities = demo_priority_functions()
    results = demo_mcts_agent()
    demo_slopes_analysis()
    priorities, points = create_comparison_visualization()
    integration_with_existing_env()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("\nExtracted modules are now available in src/:")
    print("  - src/geometry.py: QQ, Point, Line, collinearity functions")
    print("  - src/priority.py: Priority calculation functions")
    print("  - src/visualization.py: Plotting functions")
    print("  - src/algos/mcts_priority.py: MCTS with priority-based selection")
    print("\nThese can be imported and used in your RL experiments!")


if __name__ == "__main__":
    main()
