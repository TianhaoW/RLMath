import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.algos.mcts import evaluate
import numpy as np

if __name__ == "__main__":
    # Example usage

    np.random.seed(0)

    n = 100

    args = {
        'algorithm': 'MCTS',
        'n': n,
        'C': 1.41,  # 1e-7 for n=20
        'num_searches': 1_000,
        'num_workers': 14,      # >1 ⇒ parallel
        'virtual_loss': 1.0,     # magnitude to subtract at reservation
        'process_bar': True,
        'display_state': True,
        'logging_mode': False,
        'TopN': n,  # Without Priority
        "simulate_with_priority": False,
        'table_dir': f'tests/tests_mcts',  # Directory to save tables
        'figure_dir': f'tests/tests_mcts',  # Directory to save figures
    }

    evaluate(args)