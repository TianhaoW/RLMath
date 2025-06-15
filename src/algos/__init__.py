from .dqn import DQNTrainer

__all__ = ['DQNTrainer']

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