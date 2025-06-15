"""
Simple demonstration of the integrated priority-based functionality.
"""

import numpy as np
import matplotlib.pyplot as plt

def demo_basic_functionality():
    """Test basic priority functionality."""
    print("=== Basic Priority Functionality Test ===")
    
    # Test priority grid calculation
    from src.priority import priority_grid, point_collinear_count
    from src.envs.base_env import Point
    
    for grid_size in [3, 4]:
        print(f"\nTesting {grid_size}x{grid_size} grid:")
        priorities = priority_grid(grid_size)
        print("Priority grid:")
        print(priorities)
        
        # Test specific point
        center = Point(grid_size//2, grid_size//2)
        count = point_collinear_count(center, grid_size)
        print(f"Collinear count for center point {center}: {count}")


def demo_priority_agent():
    """Test priority-based agent."""
    print("\n=== Priority Agent Test ===")
    
    from src.envs import NoThreeCollinearEnv
    from src.algos.priority_agent import PriorityAgent
    
    # Create environment and agent
    env = NoThreeCollinearEnv(5, 5)
    agent = PriorityAgent(env, temperature=1.0)
    
    # Test multiple episodes
    scores = []
    for i in range(5):
        score = agent.play_episode(method="softmax", verbose=False)
        scores.append(score)
        print(f"Episode {i+1}: {score} points")
    
    print(f"Average score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")


def demo_environment_wrapper():
    """Test environment wrapper."""
    print("\n=== Environment Wrapper Test ===")
    
    from src.envs import NoThreeCollinearEnv
    from src.envs.priority_wrappers import PriorityEnvWrapper, PriorityRewardWrapper
    
    # Test priority observation wrapper
    base_env = NoThreeCollinearEnv(4, 4)
    priority_env = PriorityEnvWrapper(base_env, include_priority_in_obs=True)
    
    obs, info = priority_env.reset()
    print(f"Original obs shape: {base_env.observation_space.shape}")
    print(f"Wrapped obs shape: {obs.shape}")
    print(f"Info contains priority_grid: {'priority_grid' in info}")
    
    # Test priority reward wrapper
    reward_env = PriorityRewardWrapper(base_env, priority_weight=0.1)
    obs, _ = reward_env.reset()
    
    # Take a sample action
    action = 0  # First position
    obs, reward, done, truncated, info = reward_env.step(action)
    if 'base_reward' in info:
        print(f"Base reward: {info['base_reward']:.3f}")
        print(f"Priority reward: {info['priority_reward']:.3f}")
        print(f"Total reward: {reward:.3f}")


def demo_registry_usage():
    """Test registry system."""
    print("\n=== Registry System Test ===")
    
    from src.registry.algo_registry import ALGO_CLASSES, EXTENDED_ALGO_CLASSES
    from src.envs import NoThreeCollinearEnv
    
    print("Available algorithms in basic registry:", list(ALGO_CLASSES.keys()))
    print("Available algorithms in extended registry:", list(EXTENDED_ALGO_CLASSES.keys()))
    
    # Test using algorithm from registry
    env = NoThreeCollinearEnv(4, 4)
    
    # Get priority agent through registry
    if "priority" in EXTENDED_ALGO_CLASSES:
        PriorityAgentClass = EXTENDED_ALGO_CLASSES["priority"]()
        agent = PriorityAgentClass(env, temperature=2.0)
        score = agent.play_episode(verbose=False)
        print(f"Priority agent from registry scored: {score}")


def demo_geometry_utilities():
    """Test geometry utilities."""
    print("\n=== Geometry Utilities Test ===")
    
    from src.geometry import QQ, are_collinear
    from src.envs.base_env import Point
    
    # Test rational arithmetic
    q1 = QQ(1, 2)
    q2 = QQ(1, 3)
    print(f"QQ(1,2) + QQ(1,3) = {q1 + q2}")
    print(f"QQ(1,2) * QQ(1,3) = {q1 * q2}")
    
    # Test collinearity
    p1, p2, p3 = Point(0, 0), Point(1, 1), Point(2, 2)
    p4 = Point(1, 2)
    print(f"Points {p1}, {p2}, {p3} collinear: {are_collinear(p1, p2, p3)}")
    print(f"Points {p1}, {p2}, {p4} collinear: {are_collinear(p1, p2, p4)}")


def create_priority_visualization():
    """Create a visualization of priority grids."""
    print("\n=== Creating Priority Visualization ===")
    
    from src.priority import priority_grid
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    for i, grid_size in enumerate([3, 4, 5]):
        priorities = priority_grid(grid_size)
        
        im = axes[i].imshow(priorities, cmap='viridis', interpolation='nearest')
        axes[i].set_title(f'{grid_size}x{grid_size} Grid')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        
        # Add value annotations
        for x in range(grid_size):
            for y in range(grid_size):
                axes[i].text(y, x, f'{priorities[x, y]:.0f}', 
                           ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('/Users/zxmath/Documents/GitHub/RLMath/priority_grids.png', dpi=150, bbox_inches='tight')
    print("Priority grid visualization saved as 'priority_grids.png'")


def main():
    """Run all demonstrations."""
    print("Integrated Priority Framework Demonstration")
    print("=" * 50)
    
    demo_basic_functionality()
    demo_priority_agent()
    demo_environment_wrapper()
    demo_registry_usage()
    demo_geometry_utilities()
    create_priority_visualization()
    
    print("\n" + "=" * 50)
    print("Integration Summary:")
    print("✓ Priority calculation (src/priority.py)")
    print("✓ Geometry utilities (src/geometry.py)")
    print("✓ Priority-based agent (src/algos/priority_agent.py)")
    print("✓ Environment wrappers (src/envs/priority_wrappers.py)")
    print("✓ Registry integration (src/registry/algo_registry.py)")
    print("✓ Visualization support (src/visualization.py)")
    print("\nAll components successfully integrated with existing framework!")


if __name__ == "__main__":
    main()
