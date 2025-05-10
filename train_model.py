import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.utils import parse_config, get_logger, load_env_and_model

# ------------------------------
# Config & Logging
# ------------------------------
config = parse_config()
logger = get_logger("train", config)

train_cfg = config['train']
env_cfg = config['env']
path_cfg = config['path']

m, n = env_cfg['m'], env_cfg['n']
model_name = env_cfg['model']
env_name = env_cfg['env_type']
episodes = train_cfg['episodes']
output_dim = m * n

# ------------------------------
# Device Selection
# ------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

# ------------------------------
# Load Environment & Model
# ------------------------------
env, policy_net = load_env_and_model(config, device=device, logger=logger)
target_net = type(policy_net)(env.grid_shape, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=train_cfg['lr'])
memory = deque(maxlen=train_cfg['memory_size'])

# ------------------------------
# Training Loop
# ------------------------------
best_reward = float("-inf")
best_points = []

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        if random.random() < train_cfg['epsilon']:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_obs, reward, done, _, _ = env.step(action)
        memory.append((obs, action, reward, next_obs, done))
        obs = next_obs
        total_reward += reward

        # Training step
        if len(memory) >= train_cfg['batch_size']:
            batch = random.sample(memory, train_cfg['batch_size'])
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.stack(states), dtype=torch.float32).to(device)
            next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + train_cfg['gamma'] * next_q_values * (1 - dones)

            loss = nn.functional.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Track best run
    if total_reward > best_reward:
        best_reward = total_reward
        best_points = list(env.points)

    if episode % train_cfg['target_update_freq'] == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 100 == 0:
        logger.info(f"Episode {episode} | Reward: {total_reward} | Best: {best_reward}")
        logger.info(f"Episode ended with reward: {total_reward}, len(points): {len(env.points)}")

# ------------------------------
# Summary & Save
# ------------------------------
logger.info(f"Training complete. Best reward: {best_reward}")
if train_cfg.get("save_best_points", True):
    logger.info(f"Best point set: {best_points}")
    env.points = best_points
    env.plot()

if model_name != 'cnn':
    model_file = path_cfg['model_dir'] / f"{env_name}_{model_name}_{m}x{n}.pt"
else:
    # the cnn is for supporting the transfer learning, so there is a single file for this model with any m,n
    model_file = path_cfg['model_dir'] / f"{env_name}_{model_name}.pt"

response = input(f"Save model to {model_file}? [y/N]: ").lower()
if response == "y":
    torch.save(policy_net.state_dict(), model_file)
    logger.info(f"Model saved to: {model_file}")
else:
    logger.info("Model not saved.")
