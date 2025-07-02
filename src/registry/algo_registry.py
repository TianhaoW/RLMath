from src.algos import DQNTrainer

# Priority-based algorithms are loaded on demand to avoid circular imports
def get_priority_agent():
    from src.algos.priority_agent import PriorityAgent
    return PriorityAgent

def get_mcts_priority_agent():
    from src.algos.mcts_priority import MCTSPriorityAgent
    return MCTSPriorityAgent

def get_mcts_trainer():
    from src.algos.mcts_trainer import MCTSTrainer
    return MCTSTrainer

ALGO_CLASSES = {
    "dqn": DQNTrainer,
    "mcts": get_mcts_trainer,
}

# Extended registry with lazy loading
EXTENDED_ALGO_CLASSES = {
    "dqn": DQNTrainer,
    "priority": get_priority_agent,
    "mcts_priority": get_mcts_priority_agent,
    "mcts": get_mcts_trainer,
}