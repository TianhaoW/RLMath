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


# Convenience function for easy access to all MCTS variants
def create_mcts(m, n, variant='basic', priority_fn=None, config=None):
    """
    Create any MCTS variant with custom parameters.
    
    Args:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid
        variant (str): MCTS variant ('basic', 'priority', 'parallel', 'advanced')
        priority_fn (callable, optional): Custom priority function 
        config (dict, optional): Additional configuration parameters
    
    Returns:
        UnifiedMCTS: Configured MCTS algorithm instance
    """
    if variant == 'basic':
        return create_mcts_basic(m, n, priority_fn, config)
    elif variant == 'priority':
        return create_mcts_priority(m, n, priority_fn, config)
    elif variant == 'parallel':
        return create_mcts_parallel(m, n, priority_fn, config)
    elif variant == 'advanced':
        return create_mcts_advanced(m, n, priority_fn, config)
    else:
        raise ValueError(f"Unknown MCTS variant: {variant}")
