import torch
from src.registry.algo_registry import get_algo
from src.registry.env_registry import get_env
from src.config_utils import parse_config, get_logger

def main():
    """Train models using either DQN or MCTS algorithms."""
    config = parse_config()
    logger = get_logger("train", config)

    env_cfg = config["env"]
    m, n = env_cfg['m'], env_cfg['n']
    model_name = env_cfg.get('model', 'mcts')
    env_name = env_cfg['env_type']
    algo_name = config["algo"]["method"]
    
    logger.info(f"Training with algorithm: {algo_name}")
    logger.info(f"Environment: {env_name} ({m}x{n})")
    
    # Check if this is an MCTS algorithm
    if algo_name.startswith('mcts'):
        # MCTS algorithms don't need neural network models
        logger.info("Using MCTS algorithm - no model training required")
        
        # Get MCTS constructor from registry
        mcts_constructor = get_algo(algo_name)
        
        # Create MCTS configuration
        mcts_config = {
            'n': max(m, n),
            'num_searches': config["algo"].get("num_searches", 1000),
            'C': config["algo"].get("C", 1.414),
            'top_n': config["algo"].get("top_n", 2),
            'num_workers': config["algo"].get("num_workers", 4),
            'virtual_loss': config["algo"].get("virtual_loss", 1.0),
            'priority_type': config["algo"].get("priority_type", 'supnorm'),
            'display_state': False,
            'process_bar': True,
        }
        
        # Create MCTS instance
        mcts = mcts_constructor(mcts_config, env=None, logger=logger)
        logger.info(f"MCTS algorithm {algo_name} initialized successfully")
        
        # For MCTS, we can run evaluation/demo instead of training
        logger.info("Running MCTS evaluation...")
        # You can add evaluation logic here if needed
        
    else:
        # Traditional DQN/neural network training
        from src.config_utils import load_env_and_model
        
        # selecting the device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        env, model = load_env_and_model(config, device=device, logger=logger)
        trainer_class = get_algo(algo_name)
        trainer = trainer_class(config, env, model, device, logger)

        logger.info("Starting neural network training...")
        trainer.train()
        
        model_file = config['path']['model_dir'] / f"{env_name}_{algo_name}_{model_name}_{m}x{n}.pt"

        response = input(f"Save model to {model_file}? [y/N]: ").lower()
        if response == "y":
            trainer.model.save_checkpoint(model_file, logger)
            logger.info(f"Model saved to {model_file}")
        else:
            logger.info("Model not saved.")

if __name__ == "__main__":
    main()


