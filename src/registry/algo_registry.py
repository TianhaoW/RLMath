from src.algos import DQNTrainer, DDQNTrainer, PERDDQNTrainer
ALGO_CLASSES = {
    "dqn": DQNTrainer,
    'ddqn': DDQNTrainer,
    'ddqn_per': PERDDQNTrainer,
}