from src.envs import NoIsoscelesEnv, NoStrictIsoscelesEnv, NoThreeCollinearEnv, NoThreeCollinearEnvWithPriority

ENV_CLASSES = {
    "NoIsoscelesEnv": NoIsoscelesEnv,
    "NoStrictIsoscelesEnv": NoStrictIsoscelesEnv,
    "NoThreeCollinearEnv": NoThreeCollinearEnv,
    "NoThreeCollinearEnvWithPriority": NoThreeCollinearEnvWithPriority,
}

def get_env(env_name):
    """Get an environment class by name."""
    if env_name in ENV_CLASSES:
        return ENV_CLASSES[env_name]
    else:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(ENV_CLASSES.keys())}")