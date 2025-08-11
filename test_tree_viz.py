#!/usr/bin/env python3
"""
Test script for MCTS tree visualization functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
from src.algos.mcts import evaluate

def test_tree_visualization():
    """Test the tree visualization with a simple 3x3 grid."""
    
    args = {
        'environment': 'N3il',
        'algorithm': 'MCTS',
        'max_level_to_use_symmetry': -1,
        'n': 3,
        'C': 1.41,
        'num_searches': 50,  # Reduced for testing
        'num_workers': 1,
        'virtual_loss': 1.0,
        'process_bar': False,  # Disable progress bar for cleaner output
        'display_state': True,
        'logging_mode': True,
        'TopN': 3,
        'simulate_with_priority': False,
        'table_dir': os.path.dirname(__file__),
        'figure_dir': os.path.join(os.path.dirname(__file__), 'test_figures'),
        'random_seed': 42,
        'tree_visualization': True,  # Enable tree visualization
    }
    
    print("Starting MCTS tree visualization test...")
    print("=" * 60)
    
    # Run the evaluation
    try:
        num_points = evaluate(args)
        print(f"\nTest completed successfully!")
        print(f"Final number of points: {num_points}")
        print(f"Check the '{args['figure_dir']}' directory for visualization files.")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tree_visualization()
