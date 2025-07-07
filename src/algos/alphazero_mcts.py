"""
AlphaZero MCTS Implementation
Integrates AlphaZero features from notebooks 07_1 and 07_2 into the current MCTS structure.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, Any, Optional, List, Tuple
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm.notebook import trange

# =============================================================================
# Neural Network Architecture
# =============================================================================

class ResBlock(nn.Module):
    """Residual block for ResNet architecture."""
    
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    """ResNet architecture for AlphaZero policy and value networks."""
    
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.game = game
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(2, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

# =============================================================================
# Game Environment
# =============================================================================

class N3ilAlphaZero:
    """No-Three-In-Line environment optimized for AlphaZero training."""
    
    def __init__(self, grid_size, args):
        self.row_count = grid_size[0]
        self.column_count = grid_size[1]
        self.pts_upper_bound = np.min(grid_size) * 2
        self.action_size = self.row_count * self.column_count
        self.args = args

    def get_initial_state(self):
        """Return a grid of zeros representing an empty board."""
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action):
        """Place a point at the specified action (row-major order)."""
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = 1
        return state

    def are_collinear(self, p1, p2, p3):
        """Returns True if the three points are collinear."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

    def get_valid_moves(self, state):
        """
        Return a flattened array where 1 indicates a valid move (empty and no collinearity),
        and 0 indicates an invalid move.
        """
        valid_mask = np.zeros((self.row_count, self.column_count), dtype=np.uint8)
        coords = np.argwhere(state == 1)
        existing_pairs = list(combinations(coords, 2))

        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i, j] != 0:
                    continue
                candidate = (i, j)
                if any(self.are_collinear(p1, p2, candidate) for p1, p2 in existing_pairs):
                    continue
                valid_mask[i, j] = 1

        return valid_mask.reshape(-1)

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken):
        """
        Given a parent state and its valid move mask, return a refined valid move mask for the child.
        """
        valid = parent_valid_moves.copy()
        valid[action_taken] = 0

        existing_points = [tuple(pt) for pt in np.argwhere(parent_state == 1)]

        new_row = action_taken // self.column_count
        new_col = action_taken % self.column_count
        new_point = (new_row, new_col)

        for act in np.where(valid == 1)[0]:
            cand_row = act // self.column_count
            cand_col = act % self.column_count
            candidate = (cand_row, cand_col)

            for pt in existing_points:
                if self.are_collinear(pt, new_point, candidate):
                    valid[act] = 0
                    break

        return valid

    def get_value_and_terminated(self, state, valid_moves):
        """Return total number of points and whether a terminal condition is met."""
        if np.sum(valid_moves) > 0:
            return 0, False
        points_count = np.sum(state.reshape(-1) == 1)
        value = self.args['value_function'](points_count)
        upper_bound = self.args['value_function'](self.pts_upper_bound)
        normalized_value = value / upper_bound
        return normalized_value, True
    
    def get_encoded_state(self, state):
        """Encode state for neural network input."""
        encoded_state = np.stack(
            (state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state

    def display_state(self, state, action_prob=None):
        """Display the current grid configuration using matplotlib."""
        plt.figure(figsize=(6, 6))

        if action_prob is not None:
            assert action_prob.shape[0] == self.row_count * self.column_count, \
                f"Expected length {self.row_count * self.column_count}, got {len(action_prob)}"
            action_prob_2d = action_prob.reshape((self.row_count, self.column_count))
            flipped_probs = np.flipud(action_prob_2d)
            plt.imshow(
                flipped_probs,
                cmap='Reds',
                alpha=0.6,
                extent=[-0.5, self.column_count - 0.5, -0.5, self.row_count - 0.5],
                origin='lower',
                vmin=0, vmax=np.max(action_prob) if np.max(action_prob) > 0 else 1e-5
            )
            plt.colorbar(label="Action Probability", shrink=0.8)

        y_coords, x_coords = np.nonzero(state)
        flipped_y = self.row_count - 1 - y_coords

        plt.scatter(x_coords, flipped_y, s=100, c='blue', label='Placed Points')

        plt.grid(True)
        plt.xticks(range(self.column_count))
        plt.yticks(range(self.row_count))
        plt.xlim(-0.5, self.column_count - 0.5)
        plt.ylim(-0.5, self.row_count - 0.5)
        plt.gca().set_aspect('equal')
        plt.title("No-Three-In-Line Grid with Action Probabilities" if action_prob is not None else "No-Three-In-Line Grid")
        plt.show()

# =============================================================================
# MCTS Node with AlphaZero Features
# =============================================================================

class AlphaZeroNode:
    """MCTS Node with AlphaZero-specific features."""
    
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

        if parent is None:
            self.valid_moves = game.get_valid_moves(state)
        else:
            self.valid_moves = game.get_valid_moves_subset(
                parent.state, parent.valid_moves, self.action_taken)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        """Select best child using UCB formula."""
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child
    
    def get_ucb(self, child):
        """Calculate UCB value with AlphaZero modifications."""
        if child.visit_count == 0:
            q_value = 0
        else:
            # Use direct value instead of probability transformation
            q_value = child.value_sum / child.visit_count
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        """Expand node using policy network output."""
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)

                child = AlphaZeroNode(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        return child

    def backpropagate(self, value):
        """Backpropagate value up the tree."""
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)

# =============================================================================
# AlphaZero MCTS
# =============================================================================

class AlphaZeroMCTS:
    """MCTS implementation with AlphaZero features."""
    
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self, state):
        """Perform MCTS search with AlphaZero features."""
        root = AlphaZeroNode(self.game, self.args, state, visit_count=1)

        # Add noise and expand root
        valid_moves = self.game.get_valid_moves(state)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        # Add Dirichlet noise
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        # Apply valid moves mask
        policy *= valid_moves
        policy /= np.sum(policy) 
        root.expand(policy)

        # MCTS iterations
        for search in range(self.args['num_searches']):
            node = root

            # Selection
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.valid_moves)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                policy *= node.valid_moves
                policy /= np.sum(policy) 
                
                value = value.item()
                node = node.expand(policy)

            node.backpropagate(value)

        # Convert visit counts to action probabilities
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

# =============================================================================
# AlphaZero Training
# =============================================================================

class AlphaZero:
    """AlphaZero training implementation."""
    
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaZeroMCTS(game, args, model)
        
    def selfPlay(self):
        """Generate self-play data."""
        memory = []
        state = self.game.get_initial_state()
        
        while True:
            action_probs = self.mcts.search(state)
            memory.append((state, action_probs))

            # Apply temperature
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            state = self.game.get_next_state(state, action)

            valid_moves = self.game.get_valid_moves(state)
            value, is_terminal = self.game.get_value_and_terminated(state, valid_moves)
            
            if is_terminal:
                returnMemory = []
                for hist_state, hist_action_probs in memory:
                    hist_outcome = value
                    returnMemory.append((
                        self.game.get_encoded_state(hist_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
                
    def train(self, memory):
        """Train the neural network on self-play data."""
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx : batchIdx + self.args['batch_size']]
            if not sample:
                continue
            state, policy_targets, value_targets = zip(*sample)
            
            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        """Main training loop."""
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            if (iteration + 1) % self.args['save_interval'] == 0:
                torch.save(self.model.state_dict(), f"{self.args['weight_file_name']}_model_{iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"{self.args['weight_file_name']}_optimizer_{iteration}.pt")

# =============================================================================
# Utility Functions
# =============================================================================

def create_alphazero_config(grid_size: int, **kwargs) -> Dict[str, Any]:
    """Create default AlphaZero configuration."""
    config = {
        'C': 2,
        'num_searches': grid_size * grid_size * 10,
        'num_iterations': 3,
        'num_selfPlay_iterations': 100,
        'num_epochs': 4,
        'batch_size': 64,
        'save_interval': 1,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'value_function': lambda x: x ** 2,
        'weight_file_name': f"n3il_alphazero_{grid_size}x{grid_size}"
    }
    config.update(kwargs)
    return config

def evaluate_alphazero_model(model, game, args, num_games=10):
    """Evaluate trained AlphaZero model."""
    mcts = AlphaZeroMCTS(game, args, model)
    scores = []
    
    for _ in range(num_games):
        state = game.get_initial_state()
        num_points = 0
        
        while True:
            action_probs = mcts.search(state)
            action = np.argmax(action_probs)
            state = game.get_next_state(state, action)
            num_points += 1
            
            valid_moves = game.get_valid_moves(state)
            _, is_terminal = game.get_value_and_terminated(state, valid_moves)
            
            if is_terminal:
                break
        
        scores.append(num_points)
    
    return np.mean(scores), np.std(scores)
