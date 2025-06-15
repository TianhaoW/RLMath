from src.algos import DQNTrainer

ALGO_CLASSES = {
    "dqn": DQNTrainer,
}

# Priority-based algorithms are loaded on demand to avoid circular imports
def get_priority_agent():
    from src.algos.priority_agent import PriorityAgent
    return PriorityAgent

def get_mcts_priority_agent():
    from src.algos.mcts_priority import MCTSPriorityAgent
    return MCTSPriorityAgent

# Extended registry with lazy loading
EXTENDED_ALGO_CLASSES = {
    "dqn": DQNTrainer,
    "priority": get_priority_agent,
    "mcts_priority": get_mcts_priority_agent,
}