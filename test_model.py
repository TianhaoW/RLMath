import torch
from src.registry import ALGO_CLASSES
from src.utils import load_env_and_model, parse_config, get_logger

config = parse_config()
logger = get_logger("train", config)

env_cfg = config["env"]
m, n = env_cfg['m'], env_cfg['n']
model_name = env_cfg['model']
env_name = env_cfg['env_type']
algo_name = config["algo"]["method"]

# selecting the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

env, model = load_env_and_model(config, device=device, logger=logger)
trainer_class = ALGO_CLASSES[algo_name]
trainer = trainer_class(config, env, model, device, logger)

trainer.test()