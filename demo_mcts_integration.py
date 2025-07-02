"""
Demo script showing how to use the new integrated MCTS system.
This demonstrates the "one-click" training and evaluation functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.envs import NoThreeCollinearEnvWithPriority, Point
from src.algos.mcts_trainer import MCTSTrainer
from src.priority_functions import create_priority_function
from src.config_utils import parse_config, get_logger
import matplotlib.pyplot as plt

def demo_mcts_with_priority():
    """Demonstrate MCTS with priority function."""
    
    # Create environment with priority function
    m, n = 5, 5
    priority_fn = create_priority_function("boundary")
    env = NoThreeCollinearEnvWithPriority(m, n, priority_fn)
    
    print(f"Created {m}x{n} environment with boundary priority function")
    
    # Create MCTS configuration
    config = {
        'train': {
            'episodes': 10,  # Number of games to play
            'mcts_searches': 500,  # MCTS iterations per move
            'mcts_c_param': 1.4,
            'mcts_use_priority': True,
            'mcts_top_n': 1,
            'mcts_progress': True,
            'save_best_points': True
        }
    }
    
    # Create trainer
    trainer = MCTSTrainer(config, env, model=None, device=None, logger=None)
    
    print("Starting MCTS training...")
    results = trainer.train()
    
    print(f"\nResults:")
    print(f"Best score: {results['best_score']} points")
    print(f"Average score: {sum(results['scores']) / len(results['scores']):.2f}")
    print(f"All scores: {results['scores']}")
    
    # Show best configuration
    print(f"\nBest configuration:")
    for i, point in enumerate(results['best_points']):
        print(f"  Point {i+1}: {point}")
    
    # Visualize best result
    env.reset()
    for point in results['best_points']:
        env.self_play_add_point(point, plot=False)
    
    print("\nVisualizing best result:")
    env.plot()
    
    return results

def demo_comparison():
    """Compare different priority functions."""
    
    m, n = 4, 4
    priority_types = ["default", "boundary", "distance", "collinear_count"]
    
    results = {}
    
    for priority_type in priority_types:
        print(f"\nTesting {priority_type} priority...")
        
        priority_fn = create_priority_function(priority_type)
        env = NoThreeCollinearEnvWithPriority(m, n, priority_fn)
        
        config = {
            'train': {
                'episodes': 5,
                'mcts_searches': 200,
                'mcts_c_param': 1.4,
                'mcts_use_priority': True,
                'save_best_points': True
            }
        }
        
        trainer = MCTSTrainer(config, env, model=None, device=None, logger=None)
        result = trainer.train()
        
        results[priority_type] = {
            'best_score': result['best_score'],
            'avg_score': sum(result['scores']) / len(result['scores']),
            'scores': result['scores']
        }
        
        print(f"  Best: {result['best_score']}, Avg: {results[priority_type]['avg_score']:.2f}")
    
    # Print comparison
    print(f"\nComparison Results:")
    print(f"{'Priority Type':<15} {'Best':<6} {'Average':<8}")
    print("-" * 30)
    for priority_type, result in results.items():
        print(f"{priority_type:<15} {result['best_score']:<6} {result['avg_score']:<8.2f}")
    
    return results

def demo_config_training():
    """Demonstrate training using the config file (like train_model.py)."""
    
    print("Loading configuration...")
    config = parse_config()
    logger = get_logger("mcts_demo", config)
    
    # Load environment and trainer using the config system
    from src.config_utils import load_env_and_model
    from src.registry import ALGO_CLASSES
    
    env, model = load_env_and_model(config, device=None, logger=logger)
    
    # Get trainer class and create trainer
    algo_name = config["algo"]["method"]
    trainer_class = ALGO_CLASSES[algo_name]
    trainer = trainer_class(config, env, model, device=None, logger=logger)
    
    logger.info("Starting config-based training...")
    results = trainer.train()
    
    logger.info("Training completed!")
    
    return results

if __name__ == "__main__":
    print("=== MCTS Integration Demo ===\n")
    
    print("1. Basic MCTS with priority demo:")
    demo_results = demo_mcts_with_priority()
    
    print("\n" + "="*50 + "\n")
    
    print("2. Priority function comparison:")
    comparison_results = demo_comparison()
    
    print("\n" + "="*50 + "\n")
    
    print("3. Config-based training (like train_model.py):")
    try:
        config_results = demo_config_training()
        print("Config-based training completed successfully!")
    except Exception as e:
        print(f"Config-based training failed: {e}")
        print("This is expected if config.toml is not set up for MCTS")
    
    print("\nDemo completed!")
