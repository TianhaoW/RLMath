#!/usr/bin/env python3
"""
Test script to verify MCTS functionality with custom priority functions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algos import create_mcts
from src.envs.base_env import Point

def my_priority_func(p: Point, grid_size) -> float:
    """Custom priority function favoring corners."""
    x, y = p.x, p.y
    m, n = grid_size
    # Favor corners and edges
    return min(x, n-1-x) + min(y, m-1-y)

print("Testing MCTS functionality with custom priority...")

# Create MCTS with custom priority
config = {
    'num_searches': 100,  # Small number for quick test
    'top_n': 1,
    'c_puct': 1.0,
    'logging_mode': True  # Important for getting results
}

mcts = create_mcts(5, 5, variant='basic', priority_fn=my_priority_func, config=config)

# Test the evaluate_unified function
from src.algos.mcts_unified import evaluate_unified

result = evaluate_unified(config, variant='basic')
print(f"MCTS basic found {result} points")

# Test priority variant too
result_priority = evaluate_unified(config, variant='priority') 
print(f"MCTS priority found {result_priority} points")

print("✅ MCTS functionality test completed successfully!")
