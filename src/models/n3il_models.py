"""
Neural Network Models for No-3-In-Line
Extracted from nothreeinline-Spring-25 notebooks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_mixin import ModelMixin


class N3ILResBlock(nn.Module):
    """Residual block for N3IL networks"""
    
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


class N3ILAlphaNetwork(nn.Module, ModelMixin):
    """
    AlphaZero-style network for No-3-In-Line
    Outputs both policy and value predictions
    """
    
    def __init__(self, game, num_resBlocks=4, num_hidden=64, device='cpu'):
        super().__init__()
        self.device = device
        self.game = game
        self.num_resBlocks = num_resBlocks
        self.num_hidden = num_hidden
        
        # Input dimensions
        self.board_x = game.row_count
        self.board_y = game.column_count
        self.action_size = game.action_size
        
        # Initial convolution
        self.startBlock = nn.Sequential(
            nn.Conv2d(1, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # Residual blocks
        self.backBone = nn.ModuleList([
            N3ILResBlock(num_hidden) for _ in range(num_resBlocks)
        ])
        
        # Policy head
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.board_x * self.board_y, self.action_size)
        )
        
        # Value head
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.board_x * self.board_y, 1),
            nn.Tanh()
        )
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, board_x, board_y)
            
        Returns:
            policy_logits: Action policy logits
            value: State value estimate
        """
        x = self.startBlock(x)
        
        for resBlock in self.backBone:
            x = resBlock(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        
        return policy, value
    
    def predict(self, state):
        """
        Predict policy and value for a single state
        
        Args:
            state: Game state (numpy array)
            
        Returns:
            policy: Action probabilities
            value: State value
        """
        self.eval()
        with torch.no_grad():
            # Prepare input
            if len(state.shape) == 2:
                state = state.reshape(1, 1, state.shape[0], state.shape[1])
            elif len(state.shape) == 3:
                state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Forward pass
            policy_logits, value = self.forward(state_tensor)
            
            # Convert to probabilities
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().numpy()[0][0]
            
            return policy, value


class N3ILPolicyNetwork(nn.Module, ModelMixin):
    """
    Policy-only network for No-3-In-Line
    """
    
    def __init__(self, game, num_resBlocks=4, num_hidden=64, device='cpu'):
        super().__init__()
        self.device = device
        self.game = game
        
        # Input dimensions
        self.board_x = game.row_count
        self.board_y = game.column_count
        self.action_size = game.action_size
        
        # Network layers
        self.startBlock = nn.Sequential(
            nn.Conv2d(1, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList([
            N3ILResBlock(num_hidden) for _ in range(num_resBlocks)
        ])
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.board_x * self.board_y, self.action_size)
        )
        
        self.to(device)
    
    def forward(self, x):
        """Forward pass"""
        x = self.startBlock(x)
        
        for resBlock in self.backBone:
            x = resBlock(x)
        
        policy = self.policyHead(x)
        return policy
    
    def predict(self, state):
        """Predict policy for a single state"""
        self.eval()
        with torch.no_grad():
            if len(state.shape) == 2:
                state = state.reshape(1, 1, state.shape[0], state.shape[1])
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            policy_logits = self.forward(state_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            return policy


class N3ILValueNetwork(nn.Module, ModelMixin):
    """
    Value-only network for No-3-In-Line
    """
    
    def __init__(self, game, num_resBlocks=4, num_hidden=64, device='cpu'):
        super().__init__()
        self.device = device
        self.game = game
        
        # Input dimensions
        self.board_x = game.row_count
        self.board_y = game.column_count
        
        # Network layers
        self.startBlock = nn.Sequential(
            nn.Conv2d(1, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList([
            N3ILResBlock(num_hidden) for _ in range(num_resBlocks)
        ])
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.board_x * self.board_y, 1),
            nn.Tanh()
        )
        
        self.to(device)
    
    def forward(self, x):
        """Forward pass"""
        x = self.startBlock(x)
        
        for resBlock in self.backBone:
            x = resBlock(x)
        
        value = self.valueHead(x)
        return value
    
    def predict(self, state):
        """Predict value for a single state"""
        self.eval()
        with torch.no_grad():
            if len(state.shape) == 2:
                state = state.reshape(1, 1, state.shape[0], state.shape[1])
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            value = self.forward(state_tensor)
            
            return value.cpu().numpy()[0][0]


class N3ILSimpleFFNN(nn.Module, ModelMixin):
    """
    Simple feedforward network for No-3-In-Line
    """
    
    def __init__(self, game, hidden_layers=[128, 128], device='cpu'):
        super().__init__()
        self.device = device
        self.game = game
        
        # Input/output dimensions
        self.input_size = game.row_count * game.column_count
        self.action_size = game.action_size
        
        # Build layers
        layers = []
        prev_size = self.input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Output layers
        layers.append(nn.Linear(prev_size, self.action_size))
        
        self.network = nn.Sequential(*layers)
        self.to(device)
    
    def forward(self, x):
        """Forward pass"""
        # Flatten input
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def predict(self, state):
        """Predict action values for a single state"""
        self.eval()
        with torch.no_grad():
            if len(state.shape) == 2:
                state = state.flatten()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.forward(state_tensor)
            
            return output.cpu().numpy()[0]
