#!/usr/bin/env python3
"""
Test script for PatternBoost with environment methods
"""

from src.envs import NoThreeCollinearEnvWithPriority
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    """Test PatternBoost with environment methods"""
    print("Testing PatternBoost with environment methods")
    print("=" * 50)
    
    # Create environment
    env = NoThreeCollinearEnvWithPriority(m=46, n=46)
    print("✓ Environment created")
    
    # Test greedy search first
    print("\nTesting greedy search...")
    env.reset()
    score = env.greedy_search()
    print(f"✓ Greedy search completed with score: {score}")
    env.plot()
    
    # Test PatternBoost
    print("\nTesting PatternBoost...")
    env.reset()
    
    # PatternBoost configuration
    patternboost_config = {
        'initial_dataset_size': 2000,  # Small for testing
        'generation_size': 500,       # Small for testing
        'max_iterations': 20,         # Small for testing
        'epochs_per_iteration': 20,   # Small for testing
        'local_search_method': 'mcts_advanced', # Use greedy for local search
    }
    
    # Run PatternBoost with logging and saving
    best_score = env.patternboost(
        config=patternboost_config, 
        plot=True, 
        save_results=True,
        log_file="./logs/patternboost_test.log",
        device=device
    )
    
    print(f"✓ PatternBoost completed with best score: {best_score}")
    print("✓ All results saved to ./saved_models/")
    print("✓ Log file: ./logs/patternboost_test.log")
    print("\nPatternBoost test completed successfully!")


if __name__ == "__main__":
    main() 