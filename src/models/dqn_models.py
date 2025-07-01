# Please register your model at src/registry/model_registry.py
import re
import torch.nn as nn
import torch
from pathlib import Path
from src.models.model_mixin import RLModelMixin
import torch.nn.functional as F

# utility function for changing the path name.
def _strip_grid_suffix(path: Path) -> Path:
    """
    Removes the trailing _5x10 (grid size) from a model filename.
    """
    pattern = re.compile(r"_(\d+)x(\d+)\.pt$")
    return Path(pattern.sub(".pt", str(path)))

def _extract_grid_shape_from_filename(filename: str):
    match = re.search(r"_(\d+)x(\d+)\.pt$", filename)
    if not match:
        raise ValueError(f"Could not parse grid shape from filename: {filename}")
    m, n = int(match.group(1)), int(match.group(2))
    return m, n

class FFQNet(nn.Module, RLModelMixin):
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

    def load_from_checkpoint(self, path: Path, logger=None):
        if path.exists():
            state = torch.load(path)
            self.load_state_dict(state)
            if logger:
                logger.info(f"Loaded saved FFQNet model from {path}.")
        else:
            logger.info(f"No saved FFQNet model found at {path}. Using randomly initialized model.")

    def save_checkpoint(self, path: Path, logger=None):
        torch.save(self.state_dict(), path)
        if logger:
            logger.info(f"Saved FFQNet to {path}")

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

class ConvQNet(nn.Module, RLModelMixin):
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

    def load_from_checkpoint(self, path: Path, logger=None):
        path = _strip_grid_suffix(path)
        if path.exists():
            state = torch.load(path)
            self.load_state_dict(state)
            if logger:
                logger.info(f"Loaded saved ConvQNet from {path}")
        else:
            logger.info(f"No saved ConvQNet model found at {path}. Using randomly initialized model.")

    def save_checkpoint(self, path: Path, logger=None):
        path = _strip_grid_suffix(path)
        torch.save(self.state_dict(), path)
        if logger:
            logger.info(f"Saved ConvQNet to {path}")

###################################################################################################

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

    def _resize_pos_embed(self, old_pos_embed, old_shape, new_shape):
        """
        Interpolates ViT positional embeddings to match a new grid size.
        Accepts old_pos_embed of shape (N, D) or (1, N, D)
        Returns: shape (1, new_m * new_n, D)
        """
        if old_pos_embed.ndim == 2:
            old_pos_embed = old_pos_embed.unsqueeze(0)  # → (1, N, D)

        B, N, D = old_pos_embed.shape
        assert N == old_shape[0] * old_shape[1], "Mismatch between old pos_embed and old_shape"

        # Reshape → (B, D, H, W)
        pos_grid = old_pos_embed.transpose(1, 2).reshape(B, D, *old_shape)
        # Interpolate
        pos_grid = F.interpolate(pos_grid, size=new_shape, mode="bilinear", align_corners=False)
        # Flatten back → (1, new_m * new_n, D)
        new_pos_embed = pos_grid.flatten(2).transpose(1, 2)
        return new_pos_embed


    def load_from_checkpoint(self, path: Path, logger=None):
        stem_pattern = re.compile(r"_(\d+)x(\d+)$")
        pattern = f"{stem_pattern.sub("", str(path.stem))}_*x*.pt"
        model_dir = path.parent
        candidates = sorted(model_dir.glob(pattern))
        path_str =  str(candidates[-1]) if candidates else None  # use latest

        if path_str and Path(path_str).exists():
            old_m, old_n = _extract_grid_shape_from_filename(path_str)

            state_dict = torch.load(path_str)
            # Resize if needed
            if "pos_embed" in state_dict:
                old_pos = state_dict["pos_embed"]
                old_shape = (old_m, old_n)
                new_shape = (self.m, self.n)

                if old_shape != new_shape:
                    if logger:
                        logger.info(f"Resizing ViT pos_embed from {old_shape} → {new_shape}")
                    state_dict["pos_embed"] = self._resize_pos_embed(old_pos, old_shape, new_shape)
            self.load_state_dict(state_dict, strict=False)
            if logger:
                logger.info("ViTQNet pretrained weights loaded with resized pos_embed.")
        else:
            logger.info(f"No saved ViTQNet model found at {path}. Using randomly initialized model.")

    def save_checkpoint(self, path: Path, logger=None):
        torch.save(self.state_dict(), path)
        if logger:
            logger.info(f"Saved ViTQNet to {path}")

def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")