import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.algos.mcts import evaluate
import numpy as np

if __name__ == "__main__":
    # Example usage

    n_list = [10]

    for n in n_list:

        for i in range(1):

            args = {
                'environment': 'N3il_with_symmetry',  # Specify the environment
                'algorithm': 'MCTS',
                'n': n,
                'C': 1.41,  # 1e-7 for n=20
                'num_searches': 100*(n**2),  # Adjusted for larger n
                'num_workers': 28,      # >1 ⇒ parallel
                'virtual_loss': 1.0,     # magnitude to subtract at reservation
                'process_bar': True,
                'display_state': True,
                'logging_mode': False,
                'TopN': n,  # Without Priority
                "simulate_with_priority": False,
                'table_dir': f'tests/tests_mcts',  # Directory to save tables
                'figure_dir': f'tests/tests_mcts/figure',  # Directory to save figures
                'random_seed': i,  # Use the loop index as a seed for reproducibility
            }
            
            np.random.seed(i)
            evaluate(args)              