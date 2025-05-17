import tomllib
from pathlib import Path
import logging
import sys
from datetime import datetime
import torch
import torch.nn.functional as F

from src.registry import ENV_CLASSES, MODEL_CLASSES

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
_loggers = {}

def parse_config(config_path = CONFIG_PATH):
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    project_root = Path(__file__).resolve().parent.parent
    for key, path in config['path'].items():
        config['path'][key] = project_root / path

    # Generate timestamped run directory inside logs/
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M")
    log_root = config['path']['log_dir']
    run_log_dir = log_root / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    config['path']['run_log_dir'] = run_log_dir  # <-- new key for use elsewhere

    return config



def get_logger(name: str, config):
    """
    Returns a logger that logs to a timestamped file and to the console.
    """
    if name in _loggers:
        return _loggers[name]

    log_path = config['path']['run_log_dir'] / f"{name}.log"

    logger = logging.getLogger(name)
    # Ignore any messages below level INFO. Log INFO, WARNING, ERROR, and CRITICAL
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _loggers[name] = logger
    return logger


def resize_pos_embed(old_pos_embed, old_shape, new_shape):
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



def load_env_and_model(config, device=None, logger=None):
    env_name = config['env']['env_type']
    model_name = config['env']['model']
    m, n = config['env']['m'], config['env']['n']
    grid_shape = (m, n)
    output_dim = m * n

    # Instantiate environment
    if env_name not in ENV_CLASSES:
        raise ValueError(f"Unknown environment type: {env_name}")
    env = ENV_CLASSES[env_name](m, n)

    # Instantiate model
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model encoding: {model_name}")
    model = MODEL_CLASSES[model_name](grid_shape, output_dim)

    # Move model to device
    if device is not None:
        model = model.to(device)

    # Load model checkpoint if exists
    if model_name not in ['cnn', 'vit']:
        model_path = config['path']['model_dir'] / f"{env_name}_{model_name}_{m}x{n}.pt"
    else:
        # there is only one single cnn and vit model to handle all grid size using transfer learning.
        model_path = config['path']['model_dir'] / f"{env_name}_{model_name}.pt"

    if model_path.exists():
        # Support transfer learning for vit
        if model_name == "vit":
            # Resize pos_embed if necessary
            state_dict = torch.load(model_path, map_location=device)
            # Resize if needed
            if "pos_embed" in state_dict:
                old_pos = state_dict["pos_embed"]
                old_shape = (config['vit']['m'], config['vit']['n'])
                new_shape = (m, n)

                if old_shape != new_shape:
                    if logger:
                        logger.info(f"Resizing ViT pos_embed from {old_shape} → {new_shape}")
                    state_dict["pos_embed"] = resize_pos_embed(old_pos, old_shape, new_shape)
            model.load_state_dict(state_dict, strict=False)
            if logger:
                logger.info("Pretrained weights loaded with resized pos_embed.")
        else:
            # this is for other models
            model.load_state_dict(torch.load(model_path, map_location=device))
            if logger:
                logger.info(f"Loaded model from: {model_path}")

    else:
        if logger:
            logger.info(f"No model found at: {model_path}. Using randomly initialized model.")

    return env, model
