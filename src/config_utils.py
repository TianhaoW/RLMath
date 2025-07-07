import tomllib
from pathlib import Path
import logging
import sys
from datetime import datetime

from src.registry import ENV_CLASSES, MODEL_CLASSES
# Priority functions moved to base_env

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

def load_env_and_model(config, device=None, logger=None):
    env_name = config['env']['env_type']
    model_name = config['env']['model']
    algo_name = config["algo"]["method"]
    m, n = config['env']['m'], config['env']['n']
    grid_shape = (m, n)
    output_dim = m * n

    # Instantiate environment
    if env_name not in ENV_CLASSES:
        raise ValueError(f"Unknown environment type: {env_name}")
    
    # Check if this is a priority-based environment
    if "WithPriority" in env_name:
        priority_type = config['env'].get('priority_function', 'default')
        # Use default priority function from base_env
        priority_fn = None  # Will use default_priority from base_env
        env = ENV_CLASSES[env_name](m, n, priority_fn)
        if logger:
            logger.info(f"Created priority environment with {priority_type} priority function")
    else:
        env = ENV_CLASSES[env_name](m, n)
        if logger:
            logger.info(f"Created standard environment")

    # Instantiate model (only needed for neural network-based methods)
    model = None
    if algo_name in ['dqn'] and model_name in MODEL_CLASSES:
        model = MODEL_CLASSES[model_name](grid_shape, output_dim)
        
        # Move model to device
        if device is not None:
            model = model.to(device)

        model_path = config['path']['model_dir'] / f"{env_name}_{algo_name}_{model_name}_{m}x{n}.pt"
        model.load_from_checkpoint(model_path, logger)
        
        if logger:
            logger.info(f"Loaded model: {model_name}")
    elif logger and algo_name in ['mcts', 'mcts_priority', 'priority']:
        logger.info(f"Using model-free algorithm: {algo_name}")

    return env, model
