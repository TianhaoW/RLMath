import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.algos.base import RLAlgo

class DDQNTrainer(RLAlgo):
    def __init__(self, config, env, model, device, logger):
        super().__init__(config, env, model, device, logger)
        self.train_cfg = config["train"]
        self.env_cfg = config["env"]

        self.target_net = type(self.model)(env.grid_shape, env.grid_shape[0] * env.grid_shape[1]).to(device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_cfg["lr"])
        self.memory = deque(maxlen=self.train_cfg["memory_size"])

    def train(self):
        best_reward = float("-inf")
        best_points = []

        for episode in range(self.train_cfg["episodes"]):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                if random.random() < self.train_cfg["epsilon"]:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                        action = q_values.argmax().item()

                next_obs, reward, done, _, _ = self.env.step(action)
                self.memory.append((obs, action, reward, next_obs, done))
                obs = next_obs
                total_reward += reward

                # Update model
                if len(self.memory) >= self.train_cfg["batch_size"]:
                    self._optimize()

            if total_reward > best_reward:
                best_reward = total_reward
                best_points = list(self.env.points)

            if episode % self.train_cfg["target_update_freq"] == 0:
                self.target_net.load_state_dict(self.model.state_dict())

            if episode % 100 == 0:
                self.logger.info(f"Episode {episode} | Reward: {total_reward} | Best: {best_reward}")

        # End-of-training logging
        self.logger.info(f"Training complete. Best reward: {best_reward}")
        if self.train_cfg.get("save_best_points", True):
            self.logger.info(f"Best point set: {best_points}")
            self.env.points = best_points
            self.env.plot()

    def test(self):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()
            obs, reward, done, _, _ = self.env.step(action)
            total_reward += reward

        self.logger.info(f"Eval complete. Final reward: {total_reward}")
        self.logger.info(f"Selected points: {self.env.points}")
        self.env.plot()

    def _optimize(self):
        batch = random.sample(self.memory, self.train_cfg["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1).unsqueeze(1).to(self.device)
            target_q_values = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + self.train_cfg["gamma"] * target_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

############################################################################
# DDQN with PRE
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, td_error=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if td_error is not None:
            priority = (abs(td_error) + 1e-5) ** self.alpha
        else:
            priority = max_priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:self.pos]
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha

# DDQN with Priority Experience Replay
class PERDDQNTrainer(RLAlgo):
    def __init__(self, config, env, model, device, logger):
        super().__init__(config, env, model, device, logger)
        self.train_cfg = config["train"]
        self.env_cfg = config["env"]

        self.target_net = type(self.model)(env.grid_shape, env.grid_shape[0] * env.grid_shape[1]).to(device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_cfg["lr"])
        self.memory = PrioritizedReplayBuffer(self.train_cfg["memory_size"], alpha=self.train_cfg.get("alpha", 0.6))
        self.beta = self.train_cfg.get("beta_start", 0.4)
        self.beta_increment = (1.0 - self.beta) / self.train_cfg["episodes"]

    def train(self):
        best_reward = float("-inf")
        best_points = []

        for episode in range(self.train_cfg["episodes"]):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                if random.random() < self.train_cfg["epsilon"]:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                        action = q_values.argmax().item()

                next_obs, reward, done, _, _ = self.env.step(action)
                self.memory.add((obs, action, reward, next_obs, done))
                obs = next_obs
                total_reward += reward

                if len(self.memory.buffer) >= self.train_cfg["batch_size"]:
                    self._optimize()

            if total_reward > best_reward:
                best_reward = total_reward
                best_points = list(self.env.points)

            if episode % self.train_cfg["target_update_freq"] == 0:
                self.target_net.load_state_dict(self.model.state_dict())

            if episode % 100 == 0:
                self.logger.info(f"Episode {episode} | Reward: {total_reward} | Best: {best_reward}")

            self.beta = min(1.0, self.beta + self.beta_increment)

        self.logger.info(f"Training complete. Best reward: {best_reward}")
        if self.train_cfg.get("save_best_points", True):
            self.logger.info(f"Best point set: {best_points}")
            self.env.points = best_points
            self.env.plot()

    def test(self):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()
            obs, reward, done, _, _ = self.env.step(action)
            total_reward += reward

        self.logger.info(f"Eval complete. Final reward: {total_reward}")
        self.logger.info(f"Selected points: {self.env.points}")
        self.env.plot()

    def _optimize(self):
        batch, indices, weights = self.memory.sample(self.train_cfg["batch_size"], beta=self.beta)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            target_q_values = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + self.train_cfg["gamma"] * target_q_values * (1 - dones)

        td_errors = (q_values - targets).squeeze().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * (q_values - targets) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
