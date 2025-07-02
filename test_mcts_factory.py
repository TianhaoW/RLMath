#!/usr/bin/env python3
"""
Test script for the new MCTS factory functionality.
Verifies that MCTS can be created with custom (m, n, priority_fn) like environments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algos import create_mcts
from src.envs.base_env import Point

# Test 1: Basic MCTS without custom priority
print("Test 1: Creating basic MCTS 5x5 without custom priority...")
mcts_basic = create_mcts(5, 5, variant='basic')
print(f"✓ Basic MCTS created: {type(mcts_basic)}")

# Test 2: MCTS with custom priority function
def my_priority_func(p: Point, grid_size) -> float:
    """Custom priority function like in the colab example."""
    x, y = p.x, p.y     # gets the x,y coordinate of the point
    m, n = grid_size    # gets the size of the grids. m is number of rows, and n is number of cols
    return x + y        # define your priority score here

print("\nTest 2: Creating MCTS 5x5 with custom priority function...")
mcts_custom = create_mcts(5, 5, variant='priority', priority_fn=my_priority_func)
print(f"✓ Custom priority MCTS created: {type(mcts_custom)}")

# Test 3: Test all variants
print("\nTest 3: Creating all MCTS variants...")
variants = ['basic', 'priority', 'parallel', 'advanced']
for variant in variants:
    mcts = create_mcts(5, 5, variant=variant, priority_fn=my_priority_func)
    print(f"✓ {variant} MCTS created: {type(mcts)}")

# Test 4: Test custom grid size  
print("\nTest 4: Creating MCTS with different grid sizes...")
for m, n in [(3, 4), (6, 6), (10, 8)]:
    mcts = create_mcts(m, n, variant='basic')
    print(f"✓ MCTS created for {m}x{n} grid: {type(mcts)}")

print("\n✅ All tests passed! MCTS factory works correctly.")
print("MCTS can now be created with custom (m, n, priority_fn) just like environments.")
