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

if __name__ == "__main__":
    trainer.train()
    model_file = config['path']['model_dir'] / f"{env_name}_{algo_name}_{model_name}_{m}x{n}.pt"

    response = input(f"Save model to {model_file}? [y/N]: ").lower()
    if response == "y":
        trainer.model.save_checkpoint(model_file, logger)
    else:
        logger.info("Model not saved.")


