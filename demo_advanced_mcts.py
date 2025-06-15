#!/usr/bin/env python3
"""
Advanced MCTS Demo - Integration of luoning's advanced MCTS implementation

This script demonstrates the advanced MCTS implementation extracted from
luoning's 08_04_MCTS_along_prority_topN_Get_MCTS.ipynb notebook and 
integrated into the RLMath framework.

Features demonstrated:
1. Advanced N3il environment with priority-based move filtering
2. Numba-accelerated game logic and simulations
3. Parallel MCTS with virtual loss
4. Top-N priority selection
5. Edge preference tiebreaker
6. Comprehensive evaluation and comparison
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluation_advanced import (
    evaluate_advanced_mcts, 
    demo_single_evaluation, 
    demo_batch_comparison,
    create_default_args,
    run_batch_evaluation
)
from src.priority_advanced import ensure_priority_grid_exists
from src.algos.mcts_advanced import (
    AdvancedN3ilEnvironment,
    ParallelAdvancedMCTS,
    select_outermost_with_tiebreaker
)


def plot_results_comparison(result_lists: Dict[str, List], title: str = "MCTS Performance Comparison"):
    """
    Plot comparison of different MCTS configurations.
    
    Args:
        result_lists: Dictionary mapping configuration names to result lists
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['x', 'o', 's', '^', 'v']
    
    for i, (config_name, results) in enumerate(result_lists.items()):
        result_arr = np.array(results)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.scatter(result_arr[:, 0], result_arr[:, 1], 
                   marker=marker, color=color, label=config_name, s=100)
        
        # Add text annotations
        for x, y in result_arr:
            plt.text(x, y, f'({x}, {y})', ha='left', va='bottom', fontsize=8)
    
    plt.xlabel('Grid Size (n)')
    plt.ylabel('Number of Points Placed')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demo_advanced_features():
    """Demonstrate advanced MCTS features."""
    print("=" * 60)
    print("ADVANCED MCTS DEMO - Advanced Features")
    print("=" * 60)
    
    # 1. Show single game with visualization
    print("\n1. Single Game with Visualization (Grid 10x10, Top-3 Priority)")
    demo_single_evaluation(n=10, top_n=3, display=True)
    
    # 2. Compare different Top-N settings
    print("\n2. Comparing Different Priority Settings")
    np.random.seed(42)
    
    configs_to_test = {
        'Top-1': {'TopN': 1, 'num_searches': 5000},
        'Top-3': {'TopN': 3, 'num_searches': 5000},
        'Top-5': {'TopN': 5, 'num_searches': 5000}
    }
    
    n_list = [15, 20, 25, 30]
    comparison_results = {}
    
    for config_name, config_params in configs_to_test.items():
        print(f"\nTesting {config_name} configuration...")
        base_args = create_default_args(**config_params)
        results = run_batch_evaluation(n_list, base_args)
        comparison_results[config_name] = results
    
    plot_results_comparison(comparison_results, "Top-N Priority Comparison")
    
    # 3. Parallel vs Sequential MCTS
    print("\n3. Parallel vs Sequential MCTS Performance")
    np.random.seed(123)
    
    parallel_configs = {
        'Sequential': {'num_workers': 1, 'num_searches': 3000},
        'Parallel-2': {'num_workers': 2, 'num_searches': 3000},
        'Parallel-4': {'num_workers': 4, 'num_searches': 3000}
    }
    
    small_n_list = [10, 15, 20]
    parallel_results = {}
    
    for config_name, config_params in parallel_configs.items():
        print(f"\nTesting {config_name} configuration...")
        base_args = create_default_args(**config_params)
        results = run_batch_evaluation(small_n_list, base_args)
        parallel_results[config_name] = results
    
    plot_results_comparison(parallel_results, "Parallel vs Sequential MCTS")


def demo_integration_compatibility():
    """Demonstrate integration with existing framework."""
    print("\n" + "=" * 60)
    print("ADVANCED MCTS DEMO - Framework Integration")
    print("=" * 60)
    
    # Show how to use advanced components directly
    print("\n1. Direct Advanced Component Usage")
    
    # Create environment
    n = 8
    args = create_default_args(n=n, TopN=2, num_searches=1000)
    
    # Ensure priority grid exists
    priority_grid = ensure_priority_grid_exists(n)
    print(f"Priority grid shape: {priority_grid.shape}")
    print(f"Priority range: [{priority_grid.min():.3f}, {priority_grid.max():.3f}]")
    
    # Create environment and MCTS
    env = AdvancedN3ilEnvironment(grid_size=(n, n), args=args, priority_grid=priority_grid)
    mcts = ParallelAdvancedMCTS(env, args)
    
    # Run a few steps
    state = env.get_initial_state()
    print(f"Initial state shape: {state.shape}")
    
    for step in range(3):
        valid_moves = env.get_valid_moves(state)
        print(f"Step {step + 1}: {np.sum(valid_moves)} valid moves")
        
        mcts_probs = mcts.search(state)
        action = select_outermost_with_tiebreaker(mcts_probs, n)
        
        print(f"Selected action: {action} (row={action//n}, col={action%n})")
        state = env.get_next_state(state, action)
    
    print("Advanced integration successful!")


def demo_performance_comparison():
    """Compare with existing implementations."""
    print("\n" + "=" * 60)
    print("ADVANCED MCTS DEMO - Performance Comparison")
    print("=" * 60)
    
    print("\nRunning comprehensive batch evaluation...")
    
    # Run the batch comparison from the evaluation module
    try:
        result_top3, result_top1 = demo_batch_comparison()
        
        # Create comparison plot
        comparison_data = {
            'Advanced MCTS Top-1': result_top1,
            'Advanced MCTS Top-3': result_top3
        }
        
        plot_results_comparison(comparison_data, "Advanced MCTS Configuration Comparison")
        
        print("\nPerformance comparison completed successfully!")
        
    except Exception as e:
        print(f"Batch comparison failed: {e}")
        print("Running smaller scale test...")
        
        # Fallback to smaller test
        small_n_list = [10, 15, 20]
        base_args = create_default_args(num_searches=2000, TopN=3)
        results = run_batch_evaluation(small_n_list, base_args)
        print(f"Small scale results: {results}")


def main():
    """Main demonstration function."""
    print("ADVANCED MCTS INTEGRATION DEMO")
    print("Extracted from luoning's 08_04_MCTS_along_prority_topN_Get_MCTS.ipynb")
    print("=" * 80)
    
    try:
        # Run different demo sections
        demo_advanced_features()
        demo_integration_compatibility()
        demo_performance_comparison()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("Advanced MCTS has been successfully integrated into RLMath framework.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTrying basic functionality test...")
        try:
            # Basic test
            args = create_default_args(n=5, num_searches=100, display_state=False, logging_mode=True)
            result = evaluate_advanced_mcts(args)
            print(f"Basic test successful! Placed {result} points on 5x5 grid.")
        except Exception as e2:
            print(f"Basic test also failed: {e2}")


if __name__ == "__main__":
    main()
