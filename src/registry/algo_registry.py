from src.algos import DQNTrainer

# MCTS algorithms are loaded on demand to avoid circular imports
def get_mcts_basic():
    from src.algos.mcts_unified import UnifiedMCTS, N3ilUnified, SupNormPriority, CustomPriority
    def create_mcts(config, env=None, model=None, device=None, logger=None, m=None, n=None, priority_fn=None):
        # Support creating with custom (m, n, priority_fn) like the environment
        if m is not None and n is not None:
            grid_size = (m, n)
            config = config.copy() if config else {}
            config['n'] = n  # Keep consistent with existing API
        else:
            grid_size = (config.get('n', 5), config.get('n', 5))
        
        # Create priority system
        if priority_fn is not None:
            priority_system = CustomPriority(priority_fn, grid_size)
        else:
            priority_system = SupNormPriority()
        
        unified_env = N3ilUnified(grid_size=grid_size, args=config, priority_system=priority_system)
        return UnifiedMCTS(unified_env, config, variant='basic')
    return create_mcts

def get_mcts_priority():
    from src.algos.mcts_unified import UnifiedMCTS, N3ilUnified, SupNormPriority, CustomPriority
    def create_mcts(config, env=None, model=None, device=None, logger=None, m=None, n=None, priority_fn=None):
        # Support creating with custom (m, n, priority_fn) like the environment
        if m is not None and n is not None:
            grid_size = (m, n)
            config = config.copy() if config else {}
            config['n'] = n  # Keep consistent with existing API
        else:
            grid_size = (config.get('n', 5), config.get('n', 5))
        
        # Create priority system
        if priority_fn is not None:
            priority_system = CustomPriority(priority_fn, grid_size)
        else:
            priority_system = SupNormPriority()
        
        unified_env = N3ilUnified(grid_size=grid_size, args=config, priority_system=priority_system)
        return UnifiedMCTS(unified_env, config, variant='priority')
    return create_mcts

def get_mcts_parallel():
    from src.algos.mcts_unified import UnifiedMCTS, N3ilUnified, SupNormPriority, CustomPriority
    def create_mcts(config, env=None, model=None, device=None, logger=None, m=None, n=None, priority_fn=None):
        # Support creating with custom (m, n, priority_fn) like the environment
        if m is not None and n is not None:
            grid_size = (m, n)
            config = config.copy() if config else {}
            config['n'] = n  # Keep consistent with existing API
        else:
            grid_size = (config.get('n', 5), config.get('n', 5))
        
        # Create priority system
        if priority_fn is not None:
            priority_system = CustomPriority(priority_fn, grid_size)
        else:
            priority_system = SupNormPriority()
        
        unified_env = N3ilUnified(grid_size=grid_size, args=config, priority_system=priority_system)
        return UnifiedMCTS(unified_env, config, variant='parallel')
    return create_mcts

def get_mcts_advanced():
    from src.algos.mcts_unified import UnifiedMCTS, N3ilUnified, SupNormPriority, CustomPriority
    def create_mcts(config, env=None, model=None, device=None, logger=None, m=None, n=None, priority_fn=None):
        # Support creating with custom (m, n, priority_fn) like the environment
        if m is not None and n is not None:
            grid_size = (m, n)
            config = config.copy() if config else {}
            config['n'] = n  # Keep consistent with existing API
        else:
            grid_size = (config.get('n', 5), config.get('n', 5))
        
        # Create priority system
        if priority_fn is not None:
            priority_system = CustomPriority(priority_fn, grid_size)
        else:
            priority_system = SupNormPriority()
        
        unified_env = N3ilUnified(grid_size=grid_size, args=config, priority_system=priority_system)
        return UnifiedMCTS(unified_env, config, variant='advanced')
    return create_mcts

# Legacy MCTS trainer
def get_mcts_trainer():
    from src.algos.mcts_trainer import MCTSTrainer
    return MCTSTrainer

# Unified algorithm registry function
def get_algo(algo_name):
    """Get an algorithm class or constructor by name."""
    if algo_name in ALGO_CLASSES:
        algo = ALGO_CLASSES[algo_name]
        # If it's a function (lazy loader), call it to get the actual constructor
        if callable(algo) and hasattr(algo, '__name__') and algo.__name__.startswith('get_mcts_'):
            return algo()
        return algo
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available algorithms: {list(ALGO_CLASSES.keys())}")

ALGO_CLASSES = {
    "dqn": DQNTrainer,
    "mcts_basic": get_mcts_basic,          # Priority-Guided MCTS
    "mcts_priority": get_mcts_priority,    # Top-N Priority MCTS  
    "mcts_parallel": get_mcts_parallel,    # Advanced Parallel MCTS
    "mcts_advanced": get_mcts_advanced,    # AlphaZero-style MCTS
    "mcts": get_mcts_trainer,              # Legacy trainer for backwards compatibility
}