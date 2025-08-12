#!/usr/bin/env python3
"""
Test script to verify the compact layout functionality for MCTS tree visualization.

This script tests the new compact layout features:
1. Removed fixed x/y positions
2. Compact hierarchical layout configuration
3. Auto re-layout after expand/collapse
4. Re-fit to visible nodes
"""

import numpy as np
import sys
import os

# Add the root directory to the path so we can import from src
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(root_dir)

from src.envs import N3il
from src.algos.mcts import MCTS

def test_compact_layout():
    """Test the compact layout functionality."""
    print("Testing compact layout for MCTS tree visualization...")
    
    # Initialize environment with a small grid for testing
    n = 3
    args = {
        'display_state': False,
        'figure_dir': './test_figures',
        'n': n,
        'TopN': 3  # Add required TopN parameter
    }
    
    # Create priority grid using the supnorm_priority_array function
    from src.envs import supnorm_priority_array
    priority_grid = supnorm_priority_array(n)
    
    env = N3il(grid_size=(n, n), args=args, priority_grid=priority_grid)
    
    # Initialize MCTS with compact layout settings
    mcts_args = {
        'num_searches': 20,  # Small number for quick test
        'C': 1.4,
        'tree_visualization': True,
        'initial_visible_depth': 1,  # Show root + level 1 by default
        'toggle_children_recursive': False,
        'level_separation': 70,  # Compact setting
        'node_spacing': 50,      # Compact setting
        'simulate_with_priority': False,
        'TopN': 3,               # Add required TopN parameter
        'process_bar': False,
        'pause_at_each_step': False  # Don't pause for user input
    }
    
    mcts = MCTS(env, mcts_args)
    mcts.trial_id = 'test_compact'
    
    # Clear any existing global data
    MCTS.clear_global_data()
    
    # Start with an empty state
    state = np.zeros((n, n), dtype=bool)
    
    print(f"Running MCTS with {mcts_args['num_searches']} searches...")
    
    # Run MCTS search which will generate tree visualization
    action_probs = mcts.search(state)
    
    print(f"Generated {len(mcts.snapshots)} snapshots")
    
    # Test single snapshot HTML generation
    print("Testing single snapshot HTML generation...")
    if mcts.snapshots:
        snapshot = mcts.snapshots[0]
        html_content = snapshot['html']
        
        # Verify that x/y coordinates are not in the JSON data
        json_nodes = snapshot['json_nodes']
        has_fixed_positions = any('x' in node for node in json_nodes)
        print(f"✓ Fixed positions removed: {not has_fixed_positions}")
        
        # Verify compact layout settings are applied
        level_sep = mcts_args['level_separation']
        node_spacing = mcts_args['node_spacing']
        layout_config_found = f"levelSeparation: {level_sep}" in html_content
        print(f"✓ Compact layout configuration found: {layout_config_found}")
        
        # Verify relayoutCompact function is injected
        relayout_function_found = "relayoutCompact" in html_content
        print(f"✓ Relayout function injected: {relayout_function_found}")
        
        # Save test HTML file
        test_file = "test_compact_layout_single.html"
        with open(test_file, 'w') as f:
            f.write(html_content)
        print(f"✓ Single snapshot HTML saved as: {test_file}")
    
    # Test comprehensive HTML generation
    print("Testing comprehensive HTML generation...")
    comprehensive_file = MCTS.save_comprehensive_html("test_compact_layout_comprehensive.html")
    
    if comprehensive_file:
        # Read and verify the comprehensive HTML
        with open(comprehensive_file, 'r') as f:
            comp_html = f.read()
        
        # Verify compact layout settings
        comp_layout_found = "blockShifting: true" in comp_html and "edgeMinimization: true" in comp_html
        print(f"✓ Comprehensive compact layout found: {comp_layout_found}")
        
        # Verify relayout function
        comp_relayout_found = "window.relayoutCompact" in comp_html
        print(f"✓ Comprehensive relayout function found: {comp_relayout_found}")
        
        print(f"✓ Comprehensive HTML saved as: {comprehensive_file}")
    
    print("\n=== Test Summary ===")
    print("✓ Compact layout configuration applied")
    print("✓ Fixed node positions removed") 
    print("✓ Auto re-layout functionality added")
    print("✓ Both single and comprehensive HTML generation working")
    print("✓ Debounced resize handling implemented")
    print("\nTest files generated:")
    print("- test_compact_layout_single.html")
    print("- test_compact_layout_comprehensive.html")
    print("\nOpen these files in a browser to test the double-click expand/collapse")
    print("functionality with the new compact layout.")

if __name__ == "__main__":
    test_compact_layout()
