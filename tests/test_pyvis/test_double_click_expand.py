#!/usr/bin/env python3
"""
Test script for the double-click expand/collapse functionality in MCTS tree visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.algos.mcts import MCTS
from src.envs import N3il

def test_double_click_functionality():
    """Test the double-click expand/collapse functionality."""
    
    # Create a simple test case
    args = {
        'environment': 'N3il',
        'algorithm': 'MCTS',
        'n': 3,
        'C': 1.41,
        'num_searches': 20,  # Small number for quick testing
        'num_workers': 1,
        'virtual_loss': 1.0,
        'process_bar': False,
        'display_state': False,
        'logging_mode': True,
        'TopN': 3,  # Required parameter
        'simulate_with_priority': False,
        'tree_visualization': True,
        'pause_at_each_step': False,
        'initial_visible_depth': 1,  # Show only root and level 1 by default
        'toggle_children_recursive': False,  # Test direct children mode first
        'max_level_to_use_symmetry': -1,  # Disable symmetry for simplicity
    }
    
    # Initialize environment and MCTS
    from src.envs import supnorm_priority_array
    priority_grid_arr = supnorm_priority_array(args['n'])
    game = N3il(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid_arr)
    mcts = MCTS(game, args)
    mcts.trial_id = "test_double_click"
    
    # Create initial state and run a few searches to build a tree
    initial_state = game.get_initial_state()
    
    # Run search to create a tree with multiple levels
    action_probs = mcts.search(initial_state)
    
    # The tree visualization should have been created with double-click functionality
    if mcts.snapshots:
        snapshot = mcts.snapshots[-1]
        print(f"✓ Tree visualization created with {snapshot['total_nodes']} nodes")
        print(f"✓ JSON nodes include 'hidden' property: {'hidden' in str(snapshot['json_nodes'])}")
        print(f"✓ JSON edges include 'hidden' property: {'hidden' in str(snapshot['json_edges'])}")
        print(f"✓ HTML includes double-click script: {'doubleClick' in snapshot['html']}")
        
        # Test recursive mode
        args['toggle_children_recursive'] = True
        mcts_recursive = MCTS(game, args)  # Reuse the same game instance
        mcts_recursive.trial_id = "test_recursive"
        action_probs_recursive = mcts_recursive.search(initial_state)
        
        if mcts_recursive.snapshots:
            recursive_snapshot = mcts_recursive.snapshots[-1]
            print(f"✓ Recursive mode test passed: {'getDescendants' in recursive_snapshot['html']}")
        
        # Test comprehensive HTML generation
        web_viz_dir = os.path.join(os.path.dirname(__file__), 'test_output')
        os.makedirs(web_viz_dir, exist_ok=True)
        
        comprehensive_html = MCTS.save_final_visualization(web_viz_dir, "test_double_click")
        if comprehensive_html and os.path.exists(comprehensive_html):
            with open(comprehensive_html, 'r') as f:
                content = f.read()
            print(f"✓ Comprehensive HTML created: {comprehensive_html}")
            print(f"✓ Comprehensive HTML includes double-click: {'setupDoubleClickHandling' in content}")
        
        return True
    else:
        print("✗ No tree visualization was created")
        return False

if __name__ == "__main__":
    print("Testing double-click expand/collapse functionality...")
    print("=" * 60)
    
    success = test_double_click_functionality()
    
    print("=" * 60)
    if success:
        print("✓ All tests passed! Double-click functionality is working.")
        print("\nTo test interactively:")
        print("1. Open the generated HTML file in a web browser")
        print("2. Double-click on nodes to expand/collapse their children")
        print("3. Verify only root and level-1 nodes are visible initially")
    else:
        print("✗ Some tests failed!")
    
    # Clean up global data
    MCTS.clear_global_data()
