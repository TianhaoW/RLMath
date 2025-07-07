"""
MCTS Factory Functions
Provides convenient ways to create MCTS algorithms with custom parameters,
similar to how environments can be created with custom priority functions.
"""

from src.registry.algo_registry import get_algo


def create_mcts_basic(m, n, priority_fn=None, config=None):
    """
    Create a basic MCTS algorithm with custom grid size and priority function.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid  
        priority_fn (callable, optional): Custom priority function like in NoThreeCollinearEnvWithPriority
        config (dict, optional): Additional configuration parameters
    
    Returns:
        UnifiedMCTS: Configured MCTS algorithm instance
    """
    if config is None:
        config = {
            'num_searches': 500,
            'top_n': 1,
            'c_puct': 1.0
        }
    
    mcts_creator = get_algo('mcts_basic')
    return mcts_creator(config, m=m, n=n, priority_fn=priority_fn)


def create_mcts_priority(m, n, priority_fn=None, config=None):
    """
    Create a priority-enhanced MCTS algorithm with custom grid size and priority function.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid  
        priority_fn (callable, optional): Custom priority function like in NoThreeCollinearEnvWithPriority
        config (dict, optional): Additional configuration parameters
    
    Returns:
        UnifiedMCTS: Configured MCTS algorithm instance
    """
    if config is None:
        config = {
            'num_searches': 500,
            'top_n': 1,
            'c_puct': 1.0,
            'simulate_with_priority': True
        }
    
    mcts_creator = get_algo('mcts_priority')
    return mcts_creator(config, m=m, n=n, priority_fn=priority_fn)


def create_mcts_parallel(m, n, priority_fn=None, config=None):
    """
    Create a parallel MCTS algorithm with custom grid size and priority function.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid  
        priority_fn (callable, optional): Custom priority function like in NoThreeCollinearEnvWithPriority
        config (dict, optional): Additional configuration parameters
    
    Returns:
        UnifiedMCTS: Configured MCTS algorithm instance
    """
    if config is None:
        config = {
            'num_searches': 500,
            'top_n': 1,
            'c_puct': 1.0,
            'num_workers': 4
        }
    
    mcts_creator = get_algo('mcts_parallel')
    return mcts_creator(config, m=m, n=n, priority_fn=priority_fn)


def create_mcts_advanced(m, n, priority_fn=None, config=None):
    """
    Create an advanced MCTS algorithm with custom grid size and priority function.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid  
        priority_fn (callable, optional): Custom priority function like in NoThreeCollinearEnvWithPriority
        config (dict, optional): Additional configuration parameters
    
    Returns:
        UnifiedMCTS: Configured MCTS algorithm instance
    """
    if config is None:
        config = {
            'num_searches': 500,
            'top_n': 1,
            'c_puct': 1.0,
            'num_workers': 4,
            'simulate_with_priority': True,
            'use_annealing': True
        }
    
    mcts_creator = get_algo('mcts_advanced')
    return mcts_creator(config, m=m, n=n, priority_fn=priority_fn)


def create_alphazero_mcts(m, n, config=None):
    """
    Create an AlphaZero MCTS algorithm with custom grid size.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid
        config (dict, optional): Additional configuration parameters
    
    Returns:
        tuple: (AlphaZeroMCTS, N3ilAlphaZero, ResNet, AlphaZero) - MCTS, game, model, and trainer
    """
    from .alphazero_mcts import N3ilAlphaZero, ResNet, AlphaZero, create_alphazero_config
    import torch
    
    if config is None:
        config = create_alphazero_config(max(m, n))
    
    # Create game environment
    game = N3ilAlphaZero(grid_size=(m, n), args=config)
    
    # Create neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 4, 64, device).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create AlphaZero trainer
    alphazero = AlphaZero(model, optimizer, game, config)
    
    return alphazero.mcts, game, model, alphazero


def create_alphazero_trainer(m, n, config=None):
    """
    Create an AlphaZero trainer with custom grid size.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid
        config (dict, optional): Additional configuration parameters
    
    Returns:
        AlphaZero: Configured AlphaZero trainer instance
    """
    from .alphazero_mcts import N3ilAlphaZero, ResNet, AlphaZero, create_alphazero_config
    import torch
    
    if config is None:
        config = create_alphazero_config(max(m, n))
    
    # Create game environment
    game = N3ilAlphaZero(grid_size=(m, n), args=config)
    
    # Create neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 4, 64, device).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create AlphaZero trainer
    alphazero = AlphaZero(model, optimizer, game, config)
    
    return alphazero





# Convenience function for easy access to all MCTS variants
def create_mcts(m, n, variant='basic', priority_fn=None, config=None):
    """
    Create any MCTS variant with custom parameters.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid
        variant (str): MCTS variant ('basic', 'priority', 'parallel', 'advanced', 'alphazero')
        priority_fn (callable, optional): Custom priority function 
        config (dict, optional): Additional configuration parameters
    
    Returns:
        MCTS algorithm instance
    """
    if variant == 'basic':
        return create_mcts_basic(m, n, priority_fn, config)
    elif variant == 'priority':
        return create_mcts_priority(m, n, priority_fn, config)
    elif variant == 'parallel':
        return create_mcts_parallel(m, n, priority_fn, config)
    elif variant == 'advanced':
        return create_mcts_advanced(m, n, priority_fn, config)
    elif variant == 'alphazero':
        return create_alphazero_mcts(m, n, config)
    else:
        raise ValueError(f"Unknown MCTS variant: {variant}")
