from .dqn import DQNTrainer

# Import MCTS factory functions for easy access
from .mcts_factory import (
    create_mcts_basic,
    create_mcts_priority,
    create_mcts_parallel,
    create_mcts_advanced,
    create_mcts
)

__all__ = [
    'DQNTrainer',
    'create_mcts_basic',
    'create_mcts_priority', 
    'create_mcts_parallel',
    'create_mcts_advanced',
    'create_mcts'
]

# Lazy imports for advanced algorithms to avoid circular dependencies
def get_advanced_mcts():
    """Lazy import for advanced MCTS components."""
    from .mcts_advanced import (
        AdvancedN3ilEnvironment, 
        AdvancedMCTS, 
        ParallelAdvancedMCTS,
        select_outermost_with_tiebreaker
    )
    return {
        'AdvancedN3ilEnvironment': AdvancedN3ilEnvironment,
        'AdvancedMCTS': AdvancedMCTS,
        'ParallelAdvancedMCTS': ParallelAdvancedMCTS,
        'select_outermost_with_tiebreaker': select_outermost_with_tiebreaker
    }

def get_priority_agents():
    """Lazy import for priority-based agents."""
    from .priority_agent import PriorityGreedyAgent, PrioritySoftmaxAgent
    return {
        'PriorityGreedyAgent': PriorityGreedyAgent,
        'PrioritySoftmaxAgent': PrioritySoftmaxAgent
    }

def get_priority_mcts():
    """Lazy import for priority MCTS."""
    from .mcts_priority import PriorityMCTS, PriorityNode
    return {
        'PriorityMCTS': PriorityMCTS,
        'PriorityNode': PriorityNode
    }