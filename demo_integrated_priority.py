"""
Comprehensive demo of priority-based algorithms integrated with the existing framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import from the integrated framework
from src.envs import NoThreeCollinearEnv, PriorityEnvWrapper, PriorityRewardWrapper
from src.algos import PriorityAgent, MCTSPriorityAgent
from src.priority import priority_grid, point_collinear_count
from src.visualization import plot_no_three_in_line, plot_priority_heatmap
from src.registry import ALGO_CLASSES, ENV_CLASSES


def demo_priority_calculation():
    """Demonstrate priority calculation for different grid sizes."""
    print("=== Priority Calculation Demo ===")
    
    for grid_size in [3, 4, 5]:
        print(f"\nGrid size: {grid_size}x{grid_size}")
        priorities = priority_grid(grid_size)
        print("Priority grid:")
        print(priorities)
        
        # Find best and worst priority points
        min_idx = np.unravel_index(priorities.argmin(), priorities.shape)
        max_idx = np.unravel_index(priorities.argmax(), priorities.shape)
        print(f"Best priority point: {max_idx} (priority: {priorities[max_idx]:.2f})")
        print(f"Worst priority point: {min_idx} (priority: {priorities[min_idx]:.2f})")


def demo_priority_agent():
    """Demonstrate the priority-based agent."""
    print("\n=== Priority Agent Demo ===")
    
    # Create environment
    env = NoThreeCollinearEnv(m=5, n=5)
    
    # Test different agent configurations
    configs = [
        {"noise": 0.0, "temperature": 1.0, "name": "Deterministic"},
        {"noise": 0.0, "temperature": 2.0, "name": "High Temperature"},
        {"noise": 0.1, "temperature": 1.0, "name": "With Noise"}
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} Agent ---")
        agent = PriorityAgent(env, noise=config["noise"], temperature=config["temperature"])
        
        # Evaluate agent
        results = agent.evaluate_agent(num_episodes=5, method="softmax")
        print(f"Mean score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
        print(f"Score range: {results['min_score']} - {results['max_score']}")


def demo_environment_wrappers():
    """Demonstrate environment wrappers with priority features."""
    print("\n=== Environment Wrappers Demo ===")
    
    # Create base environment
    base_env = NoThreeCollinearEnv(m=4, n=4)
    
    # Test priority observation wrapper
    print("\n--- Priority Observation Wrapper ---")
    priority_obs_env = PriorityEnvWrapper(base_env, include_priority_in_obs=True)
    obs, info = priority_obs_env.reset()
    print(f"Original observation shape: {base_env.observation_space.shape}")
    print(f"Wrapped observation shape: {obs.shape}")
    print(f"Priority grid available in info: {'priority_grid' in info}")
    
    # Test priority reward wrapper
    print("\n--- Priority Reward Wrapper ---")
    priority_reward_env = PriorityRewardWrapper(base_env, priority_weight=0.1)
    obs, _ = priority_reward_env.reset()
    
    # Take a few actions to see reward modification
    for i in range(3):
        valid_actions = []
        for action in range(base_env.action_space.n):
            point = base_env.decode_action(action)
            if not base_env.is_selected(point):
                valid_actions.append(action)
        
        if valid_actions:
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = priority_reward_env.step(action)
            if 'base_reward' in info:
                print(f"Action {action}: Base reward: {info['base_reward']:.3f}, "
                      f"Priority reward: {info['priority_reward']:.3f}, "
                      f"Total: {reward:.3f}")
        
        if done:
            break


def demo_registry_integration():
    """Demonstrate integration with the registry system."""
    print("\n=== Registry Integration Demo ===")
    
    # Show available algorithms
    print("Available algorithms:")
    for name, algo_class in ALGO_CLASSES.items():
        print(f"  - {name}: {algo_class.__name__}")
    
    # Create environment and algorithm through registry
    env = NoThreeCollinearEnv(m=4, n=4)
    
    # Test priority agent from registry
    if "priority" in ALGO_CLASSES:
        PriorityAgentClass = ALGO_CLASSES["priority"]
        agent = PriorityAgentClass(env, temperature=1.5)
        score = agent.play_episode(method="softmax", verbose=False)
        print(f"\nPriority agent (from registry) score: {score}")


def demo_stable_baselines_integration():
    """Demonstrate how to use priority features with Stable Baselines3."""
    print("\n=== Stable Baselines3 Integration Demo ===")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        
        # Create environment with priority features
        base_env = NoThreeCollinearEnv(m=4, n=4)
        env = PriorityRewardWrapper(base_env, priority_weight=0.1)
        
        # Check environment compatibility
        print("Checking environment compatibility with SB3...")
        check_env(env, warn=True)
        print("Environment check passed!")
        
        # Create and train a simple PPO model
        print("Training PPO model with priority rewards...")
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
        model.learn(total_timesteps=1000)
        
        # Test the model
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done:
                break
        
        print(f"PPO model total reward: {total_reward:.2f}")
        
    except ImportError:
        print("Stable Baselines3 not available. Skipping this demo.")


def visualize_priority_comparison():
    """Visualize priority distributions for different grid sizes."""
    print("\n=== Priority Visualization ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, grid_size in enumerate([3, 4, 5]):
        priorities = priority_grid(grid_size)
        
        im = axes[i].imshow(priorities, cmap='viridis', interpolation='nearest')
        axes[i].set_title(f'{grid_size}x{grid_size} Priority Grid')
        axes[i].set_xlabel('X coordinate')
        axes[i].set_ylabel('Y coordinate')
        
        # Add text annotations
        for x in range(grid_size):
            for y in range(grid_size):
                axes[i].text(y, x, f'{priorities[x, y]:.1f}', 
                           ha='center', va='center', color='white', fontsize=8)
        
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('/Users/zxmath/Documents/GitHub/RLMath/priority_comparison.png', dpi=150, bbox_inches='tight')
    print("Priority comparison saved as 'priority_comparison.png'")
    # plt.show()


def main():
    """Run all demonstrations."""
    print("Comprehensive Demo of Priority-Based Framework Integration")
    print("=" * 60)
    
    # Run demonstrations
    demo_priority_calculation()
    demo_priority_agent()
    demo_environment_wrappers()
    demo_registry_integration()
    demo_stable_baselines_integration()
    
    # Generate visualizations
    visualize_priority_comparison()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("\nThe framework now includes:")
    print("- Priority calculation utilities (src/priority.py)")
    print("- Geometry utilities with exact arithmetic (src/geometry.py)")
    print("- Priority-based agents (src/algos/priority_agent.py)")
    print("- MCTS with priority (src/algos/mcts_priority.py)")
    print("- Environment wrappers (src/envs/priority_wrappers.py)")
    print("- Registry integration (src/registry/algo_registry.py)")
    print("- Visualization utilities (src/visualization.py)")


if __name__ == "__main__":
    main()
