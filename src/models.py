# Please register your model at src/registry/model_registry.py
import torch.nn as nn

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

class ConvQNet(nn.Module):
    def __init__(self, grid_shape, output_dim):
        super().__init__()
        m, n = grid_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.q_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, m, n) → (B, 1, m, n)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        q_map = self.q_head(x)  # (B, 1, m, n)
        return q_map.view(x.size(0), -1)  # → (B, m*n)
