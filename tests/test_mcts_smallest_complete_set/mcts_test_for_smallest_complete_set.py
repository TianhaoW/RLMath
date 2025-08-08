# we used exponential reward function

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.algos.mcts import evaluate

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MCTS tests for a range of n values.")
    parser.add_argument("--start", type=int, default=3, help="Starting value of n (inclusive)")
    parser.add_argument("--end", type=int, default=101, help="Ending value of n (exclusive)")
    parser.add_argument("--step", type=int, default=10, help="Step size for n values")
    parser.add_argument("--repeat", type=int, default=10, help="Number of runs for each n value")
    args_cli = parser.parse_args()

    # Generate list of n values
    n_list = range(args_cli.start, args_cli.end, args_cli.step)


    for n in n_list:

        for i in range(args_cli.repeat):

            args = {
                'environment': 'N3il_with_symmetry',  # Specify the environment
                'algorithm': 'MCTS',
                'max_level_to_use_symmetry': 1,  # -1 means no symmetry
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
                'table_dir': os.path.dirname(__file__),  # Directory to save tables
                'figure_dir': os.path.join(os.path.dirname(__file__), 'figure'),  # Directory to save figures
                'random_seed': i,  # Use the loop index as a seed for reproducibility
            }
            
            # np.random.seed(i)  # Removed: seed is now handled via args['random_seed']
            evaluate(args)              