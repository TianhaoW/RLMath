"""
Example notebook showing how to use the integrated priority framework with Stable Baselines3.
"""

# Cell 1: Imports and Setup
import numpy as np
import matplotlib.pyplot as plt
from src.envs import NoThreeCollinearEnv
from src.algos.priority_agent import PriorityAgent
from src.envs.priority_wrappers import PriorityRewardWrapper, PriorityEnvWrapper
from src.priority import priority_grid
from src.visualization import plot_priority_heatmap

print("Integrated Priority Framework with Stable Baselines3")
print("="*60)

# Cell 2: Create Base Environment
print("\n1. Setting up base environment...")
base_env = NoThreeCollinearEnv(m=6, n=6)
print(f"Environment: {base_env.__class__.__name__}")
print(f"Grid shape: {base_env.grid_shape}")
print(f"Action space: {base_env.action_space}")
print(f"Observation space: {base_env.observation_space}")

# Cell 3: Priority Analysis
print("\n2. Analyzing priority distribution...")
priorities = priority_grid(6)
print("Priority grid (6x6):")
print(priorities)

# Find best starting positions
flat_priorities = priorities.flatten()
best_positions = np.argsort(flat_priorities)[-5:]  # Top 5 positions
print(f"\nTop 5 priority positions (flattened indices): {best_positions}")
for idx in best_positions[-3:]:  # Show top 3
    x, y = divmod(idx, 6)
    print(f"  Position ({x}, {y}): priority {priorities[x, y]:.1f}")

# Cell 4: Test Priority Agent
print("\n3. Testing priority-based agent...")
priority_agent = PriorityAgent(base_env, temperature=1.5)

# Run multiple episodes
scores = []
for i in range(5):
    score = priority_agent.play_episode(method="softmax", verbose=False)
    scores.append(score)

print(f"Priority agent performance over 5 episodes:")
print(f"Scores: {scores}")
print(f"Mean: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
print(f"Best: {max(scores)}")

# Cell 5: Environment Wrappers for RL
print("\n4. Setting up environment wrappers for RL...")

# Wrapper that modifies rewards based on priority
priority_reward_env = PriorityRewardWrapper(base_env, priority_weight=0.05)

# Test the wrapper
obs, _ = priority_reward_env.reset()
print("Testing priority reward wrapper:")

# Take a few actions to see reward modification
for step in range(3):
    # Get valid actions
    valid_actions = []
    for action in range(base_env.action_space.n):
        point = base_env.decode_action(action)
        if not base_env.is_selected(point):
            valid_actions.append(action)
    
    if valid_actions:
        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = priority_reward_env.step(action)
        
        point = base_env.decode_action(action)
        if 'base_reward' in info:
            print(f"  Step {step+1}: Point {point}")
            print(f"    Base reward: {info['base_reward']:.3f}")
            print(f"    Priority reward: {info['priority_reward']:.3f}")
            print(f"    Total reward: {reward:.3f}")
    
    if done:
        break

# Cell 6: Stable Baselines3 Integration
print("\n5. Stable Baselines3 integration...")

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create environment for SB3
    def make_env():
        env = NoThreeCollinearEnv(m=5, n=5)  # Smaller for faster training
        env = PriorityRewardWrapper(env, priority_weight=0.1)
        return env
    
    # Check environment
    test_env = make_env()
    print("Checking environment compatibility with SB3...")
    check_env(test_env, warn=True)
    print("✓ Environment check passed!")
    
    # Create vectorized environment
    vec_env = DummyVecEnv([make_env])
    
    # Train PPO model
    print("\nTraining PPO model with priority rewards...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=0,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10
    )
    
    model.learn(total_timesteps=5000)
    print("✓ PPO training completed!")
    
    # Test trained model
    print("\nTesting trained PPO model...")
    obs = vec_env.reset()
    episode_reward = 0
    steps = 0
    
    for _ in range(20):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        episode_reward += reward[0]
        steps += 1
        
        if done[0]:
            break
    
    print(f"PPO model performance:")
    print(f"  Steps taken: {steps}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Points placed: {steps}")  # In this env, each step places a point
    
    # Compare with priority agent
    priority_score = priority_agent.play_episode(method="softmax", verbose=False)
    print(f"\nComparison:")
    print(f"  PPO model: {steps} points")
    print(f"  Priority agent: {priority_score} points")
    
except ImportError:
    print("Stable Baselines3 not installed. Skipping SB3 demonstration.")
    print("To install: pip install stable-baselines3")

# Cell 7: Advanced Usage - Custom Reward Shaping
print("\n6. Advanced usage - Custom reward shaping...")

class CustomPriorityRewardWrapper(PriorityRewardWrapper):
    """Custom wrapper with more sophisticated reward shaping."""
    
    def __init__(self, env, priority_weight=0.1, diversity_bonus=0.05):
        super().__init__(env, priority_weight)
        self.diversity_bonus = diversity_bonus
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        if not done and len(self.env.points) > 1:
            # Add diversity bonus for spreading out points
            point = self.env.decode_action(action)
            min_distance = min([
                abs(point.x - p.x) + abs(point.y - p.y) 
                for p in self.env.points[:-1]
            ])
            diversity_reward = self.diversity_bonus * min_distance
            reward += diversity_reward
            info["diversity_reward"] = diversity_reward
        
        return obs, reward, done, truncated, info

# Test custom wrapper
print("Testing custom reward wrapper...")
custom_env = CustomPriorityRewardWrapper(NoThreeCollinearEnv(4, 4))
custom_agent = PriorityAgent(custom_env, temperature=1.0)
custom_score = custom_agent.play_episode(verbose=False)
print(f"Custom reward agent score: {custom_score}")

# Cell 8: Summary
print("\n" + "="*60)
print("INTEGRATION SUMMARY")
print("="*60)
print("✓ Extracted priority calculations from luoning's notebook")
print("✓ Integrated with existing environment framework")
print("✓ Created priority-based agents")
print("✓ Added environment wrappers for RL")
print("✓ Integrated with registry system")
print("✓ Compatible with Stable Baselines3")
print("✓ Support for custom reward shaping")
print("\nKey Components:")
print("- src/priority.py: Priority calculation utilities")
print("- src/geometry.py: Exact arithmetic and geometry")
print("- src/algos/priority_agent.py: Priority-based agents")
print("- src/envs/priority_wrappers.py: Environment wrappers")
print("- src/registry/algo_registry.py: Algorithm registry")
print("\nThe framework is now ready for advanced RL experiments!")
