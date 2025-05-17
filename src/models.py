# Please register your model at src/registry/model_registry.py
import torch.nn as nn
import torch

class FFQNet(nn.Module):
    def __init__(self, grid_shape, output_dim):
        super().__init__()
        m, n = grid_shape
        self.input_dim = m * n
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x: (B, m, n) → flatten to (B, m*n)
        x = x.view(x.size(0), -1)
        return self.model(x)

###################################################################################################

# This is a residual block used in the following ConvQNet
class ResidualBlock(nn.Module):
    def __init__(self, channels, ker_size=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ker_size, padding=(ker_size - 1) // 2),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        return self.relu(out + x)  # residual connection

class ConvQNet(nn.Module):
    def __init__(self, grid_shape, output_dim):
        super().__init__()
        m, n = grid_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 51, padding=25),
            nn.ReLU(),
            ResidualBlock(64, ker_size = 5),
            # ResidualBlock(64, ker_size=51),
        )

        self.q_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, m, n) → (B, 1, m, n)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        q_map = self.q_head(x)  # (B, 1, m, n)
        return q_map.view(x.size(0), -1)  # → (B, m*n)

###################################################################################################

import torch
import torch.nn as nn

class ViTQNet(nn.Module):
    def __init__(self, grid_shape, output_dim, embed_dim=64, num_heads=4, num_layers=4):
        """
        Args:
            grid_shape: (m, n)
            output_dim: should be m * n
            embed_dim: dimensionality of token embeddings
            num_heads: number of attention heads
            num_layers: number of transformer encoder layers
        """
        super().__init__()
        m, n = grid_shape
        self.m, self.n = m, n
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        # CNN stem to embed local features
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1)

        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, m * n, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True  # Use (B, seq_len, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Q-value projection
        self.q_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, m, n = x.shape
        assert (m, n) == (self.m, self.n), "Input grid size must match init shape"

        x = x.unsqueeze(1)                          # (B, 1, m, n)
        x = self.patch_embed(x)                     # (B, D, m, n)
        x = x.flatten(2).transpose(1, 2)            # (B, m*n, D)
        x = x + self.pos_embed                      # Add positional embeddings

        x = self.transformer(x)                     # (B, m*n, D)
        q_vals = self.q_head(x).squeeze(-1)         # (B, m*n)
        return q_vals
