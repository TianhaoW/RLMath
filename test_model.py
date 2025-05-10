import torch
from src.utils import parse_config, get_logger, load_env_and_model

# ------------------------------
# Config & Logger
# ------------------------------
config = parse_config()
logger = get_logger("test", config)

env_cfg = config['env']
path_cfg = config['path']

m, n = env_cfg['m'], env_cfg['n']
model_name = env_cfg['model']
env_name = env_cfg['env_type']
output_dim = m * n

# ------------------------------
# Device Selection
# ------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

# ------------------------------
# Load Env + Model
# ------------------------------
env, model = load_env_and_model(config, device=device, logger=logger)
model.eval()

# ------------------------------
# Greedy Evaluation Loop
# ------------------------------
obs, _ = env.reset()
done = False

while not done:
    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = q_values.argmax().item()
    obs, reward, done, _, _ = env.step(action)

total_reward = sum([1 for _ in env.points])
logger.info(f"Evaluation complete.")
logger.info(f"Final reward: {total_reward}")
logger.info(f"Final point set: {env.points}")
env.plot()
