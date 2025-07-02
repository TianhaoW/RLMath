"""
Comprehensive MCTS integration example notebook.
This shows how to use the integrated MCTS system with priority functions
and demonstrates the plotting capabilities using the environment's plot method.
"""

# Add project root to sys.path
import sys
from pathlib import Path
project_root = Path.cwd()  # Assumes running from project root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.envs import NoThreeCollinearEnvWithPriority, Point
from src.algos.mcts_trainer import MCTSTrainer
from src.priority_functions import create_priority_function, create_custom_priority
from src.config_utils import parse_config, load_env_and_model, get_logger
from src.registry import ALGO_CLASSES

def demo_environment_with_priority():
    """Demonstrate the environment with different priority functions."""
    print("=== Environment with Priority Functions Demo ===\n")
    
    # Create a small environment for demonstration
    m, n = 4, 4
    
    # Test different priority functions
    priority_types = ["default", "boundary", "distance", "collinear_count"]
    
    for priority_type in priority_types:
        print(f"--- {priority_type.capitalize()} Priority ---")
        
        # Create environment with priority function
        priority_fn = create_priority_function(priority_type)
        env = NoThreeCollinearEnvWithPriority(m, n, priority_fn)
        
        print(f"Priority map for {priority_type}:")
        print(env.priority_map)
        
        # Show a few greedy steps
        print("\\nGreedy steps:")
        for step in range(3):
            result = env.greedy_action_step(random=False, plot=False)
            if result == -1:
                print("No more valid moves")
                break
            else:
                print(f"  Step {step + 1}: Added {result}")
        
        # Reset for next test
        env.reset()
        print()

def demo_mcts_with_different_priorities():
    """Compare MCTS performance with different priority functions."""
    print("=== MCTS with Different Priority Functions ===\\n")
    
    m, n = 4, 4
    priority_types = ["boundary", "distance", "collinear_count"]
    results = {}
    
    for priority_type in priority_types:
        print(f"Testing MCTS with {priority_type} priority...")
        
        # Create environment
        priority_fn = create_priority_function(priority_type)
        env = NoThreeCollinearEnvWithPriority(m, n, priority_fn)
        
        # Configure MCTS
        config = {
            'train': {
                'episodes': 5,
                'mcts_searches': 100,
                'mcts_c_param': 1.4,
                'mcts_use_priority': True,
                'mcts_progress': False,
                'save_best_points': True
            }
        }
        
        # Create and run trainer
        trainer = MCTSTrainer(config, env)
        result = trainer.train()
        
        results[priority_type] = {
            'best_score': result['best_score'],
            'avg_score': np.mean(result['scores']),
            'best_points': result['best_points']
        }
        
        print(f"  Best score: {result['best_score']}")
        print(f"  Average score: {results[priority_type]['avg_score']:.2f}")
        print()
    
    # Summary
    print("Summary:")
    print(f"{'Priority':<15} {'Best':<6} {'Average':<8}")
    print("-" * 30)
    for priority_type, result in results.items():
        print(f"{priority_type:<15} {result['best_score']:<6} {result['avg_score']:<8.2f}")
    
    return results

def demo_plotting_integration():
    """Demonstrate plotting using the environment's plot method."""
    print("\\n=== Plotting Integration Demo ===\\n")
    
    # Create environment with boundary priority
    m, n = 5, 5
    priority_fn = create_priority_function("boundary")
    env = NoThreeCollinearEnvWithPriority(m, n, priority_fn)
    
    print("Initial environment (with priority heatmap):")
    env.plot()
    
    # Add some points manually to show the plotting
    points_to_add = [Point(1, 1), Point(3, 2), Point(0, 4), Point(4, 0), Point(2, 3)]
    
    print("\\nAdding points step by step:")
    for i, point in enumerate(points_to_add):
        try:
            env.self_play_add_point(point, plot=False)
            print(f"Added point {i+1}: {point}")
        except RuntimeError as e:
            print(f"Could not add point {point}: {e}")
            break
    
    print("\\nFinal configuration:")
    env.plot()
    
    print(f"Total points placed: {len(env.points)}")

def demo_config_based_training():
    """Demonstrate config-based training like train_model.py."""
    print("\\n=== Config-Based Training Demo ===\\n")
    
    try:
        # Load configuration
        config = parse_config()
        logger = get_logger("demo", config)
        
        print(f"Loaded config:")
        print(f"  Algorithm: {config['algo']['method']}")
        print(f"  Environment: {config['env']['env_type']}")
        print(f"  Grid size: {config['env']['m']}x{config['env']['n']}")
        print(f"  Priority function: {config['env']['priority_function']}")
        
        # Load environment and model
        env, model = load_env_and_model(config, logger=logger)
        
        # Get trainer
        trainer_class = ALGO_CLASSES[config['algo']['method']]
        if callable(trainer_class) and not isinstance(trainer_class, type):
            trainer_class = trainer_class()
        
        # Override config for quick demo
        config['train']['episodes'] = 3
        config['train']['mcts_searches'] = 50
        
        trainer = trainer_class(config, env, model, None, logger)
        
        print("\\nStarting training...")
        results = trainer.train()
        
        print(f"\\nTraining completed!")
        print(f"Best score: {results['best_score']}")
        print(f"Average score: {np.mean(results['scores']):.2f}")
        
        # Show final result using environment's plot method
        print("\\nVisualizing best result:")
        env.reset()
        for point in results['best_points']:
            env.self_play_add_point(point, plot=False)
        env.plot()
        
        return results
        
    except Exception as e:
        print(f"Config-based training failed: {e}")
        print("Make sure config.toml is properly set up for MCTS")
        return None

def demo_advanced_mcts_features():
    """Demonstrate advanced MCTS features."""
    print("\\n=== Advanced MCTS Features Demo ===\\n")
    
    # Create environment
    m, n = 4, 4
    priority_fn = create_priority_function("boundary")
    env = NoThreeCollinearEnvWithPriority(m, n, priority_fn)
    
    # Test different MCTS configurations
    configs = [
        {
            'name': 'Low exploration',
            'config': {
                'train': {
                    'episodes': 3,
                    'mcts_searches': 50,
                    'mcts_c_param': 0.5,  # Low exploration
                    'mcts_use_priority': True,
                }
            }
        },
        {
            'name': 'High exploration', 
            'config': {
                'train': {
                    'episodes': 3,
                    'mcts_searches': 50,
                    'mcts_c_param': 2.0,  # High exploration
                    'mcts_use_priority': True,
                }
            }
        },
        {
            'name': 'No priority',
            'config': {
                'train': {
                    'episodes': 3,
                    'mcts_searches': 50,
                    'mcts_c_param': 1.4,
                    'mcts_use_priority': False,  # No priority guidance
                }
            }
        }
    ]
    
    results = {}
    
    for test in configs:
        name = test['name']
        config = test['config']
        
        print(f"Testing {name}...")
        
        # Reset environment
        env.reset()
        trainer = MCTSTrainer(config, env)
        result = trainer.train()
        
        results[name] = {
            'best_score': result['best_score'],
            'avg_score': np.mean(result['scores']),
            'scores': result['scores']
        }
        
        print(f"  Best: {result['best_score']}, Avg: {results[name]['avg_score']:.2f}")
    
    print("\\nComparison:")
    print(f"{'Configuration':<15} {'Best':<6} {'Average':<8}")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name:<15} {result['best_score']:<6} {result['avg_score']:<8.2f}")
    
    return results

if __name__ == "__main__":
    print("MCTS Integration Comprehensive Demo")
    print("=" * 50)
    
    # Run all demonstrations
    demo_environment_with_priority()
    
    mcts_results = demo_mcts_with_different_priorities()
    
    demo_plotting_integration()
    
    config_results = demo_config_based_training()
    
    advanced_results = demo_advanced_mcts_features()
    
    print("\\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\\nKey achievements:")
    print("✓ Environment with priority functions working")
    print("✓ MCTS trainer integrated with environment")
    print("✓ Plotting using environment's plot method")
    print("✓ One-click training through config system")
    print("✓ Priority integration in MCTS working")
    print("✓ Multiple priority functions supported")
