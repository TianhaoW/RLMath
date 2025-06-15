"""
Advanced MCTS evaluation and gameplay functions.
Extracted from luoning's 08_04_MCTS_along_prority_topN_Get_MCTS.ipynb
"""

import time
import random
import numpy as np
from typing import Dict, Any, Optional

from .algos.mcts_advanced import (
    AdvancedN3ilEnvironment, 
    AdvancedMCTS, 
    ParallelAdvancedMCTS, 
    select_outermost_with_tiebreaker
)
from .priority_advanced import load_priority_grid, ensure_priority_grid_exists


def evaluate_advanced_mcts(args: Dict[str, Any]) -> Optional[int]:
    """
    Evaluate advanced MCTS performance on No-Three-in-Line problem.
    
    Args:
        args (Dict[str, Any]): Configuration dictionary containing:
            - n (int): Grid size
            - C (float): UCB exploration parameter
            - num_searches (int): Number of MCTS simulations
            - num_workers (int): Number of parallel workers (>1 for parallel MCTS)
            - virtual_loss (float): Virtual loss magnitude for parallel MCTS
            - process_bar (bool): Show progress bar
            - display_state (bool): Display board states
            - logging_mode (bool): Return result for logging
            - TopN (int): Number of top priority levels to consider
            - priority_grid_dir (str): Directory for priority grids
    
    Returns:
        Optional[int]: Number of points placed if logging_mode=True, None otherwise
    """
    # Load or compute priority grid
    n = args['n']
    priority_grid_dir = args.get('priority_grid_dir', 'priority_grids')
    priority_grid_arr = ensure_priority_grid_exists(n, priority_grid_dir)
    
    start = time.time()
    
    # Create environment with priority grid
    n3il = AdvancedN3ilEnvironment(grid_size=(n, n), args=args, priority_grid=priority_grid_arr)

    # Choose MCTS implementation based on worker count
    mcts_cls = ParallelAdvancedMCTS if args.get('num_workers', 1) > 1 else AdvancedMCTS
    mcts = mcts_cls(n3il, args=args)

    state = n3il.get_initial_state()
    num_of_points = 0
    mcts_probs = None

    while True:
        if args.get('display_state', False):
            print("---------------------------")
            print(f"Number of points: {num_of_points}")
            print(state)

        valid_moves = n3il.get_valid_moves(state)
        value, is_terminal = n3il.get_value_and_terminated(state, valid_moves)

        if is_terminal:
            print("*******************************************************************")
            print(f"Trial Terminated with {num_of_points} points. Final valid configuration:")
            if args.get('display_state', False):
                print(state)
                if mcts_probs is not None:
                    n3il.display_state(state, mcts_probs)
            end = time.time()
            print(f"Time: {end - start:.6f} sec")
            break

        # Get MCTS probabilities
        mcts_probs = mcts.search(state)

        # Use outermost-priority selector to pick action
        action = select_outermost_with_tiebreaker(mcts_probs, n)

        # Display MCTS probabilities and board
        if args.get('display_state', False):
            n3il.display_state(state, mcts_probs)

        # Apply action
        num_of_points += 1
        state = n3il.get_next_state(state, action)
    
    if args.get('logging_mode', False):
        return num_of_points
    return None


def run_batch_evaluation(n_list: list, base_args: Dict[str, Any]) -> list:
    """
    Run batch evaluation across multiple grid sizes.
    
    Args:
        n_list (list): List of grid sizes to evaluate
        base_args (Dict[str, Any]): Base configuration arguments
        
    Returns:
        list: List of (grid_size, num_points) tuples
    """
    result_list = []
    
    for n in n_list:
        args = base_args.copy()
        args['n'] = n
        args['logging_mode'] = True
        args['display_state'] = False
        args['process_bar'] = False
        
        number_of_points = evaluate_advanced_mcts(args)
        print(f"Grid Size n={n} | # of points: {number_of_points}")
        result_list.append((n, number_of_points))
    
    return result_list


def create_default_args(**overrides) -> Dict[str, Any]:
    """
    Create default arguments for MCTS evaluation with optional overrides.
    
    Args:
        **overrides: Keyword arguments to override defaults
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    default_args = {
        'n': 30,
        'C': 0.2,
        'num_searches': 10_000,
        'num_workers': 4,      # >1 => parallel
        'virtual_loss': 1.0,   # magnitude to subtract at reservation
        'process_bar': True,
        'display_state': True,
        'logging_mode': False,
        'TopN': 3,
        'priority_grid_dir': 'priority_grids'
    }
    
    default_args.update(overrides)
    return default_args


def demo_single_evaluation(n: int = 30, top_n: int = 3, display: bool = True):
    """
    Run a single demonstration of advanced MCTS.
    
    Args:
        n (int): Grid size
        top_n (int): Number of top priority levels to consider
        display (bool): Whether to display the game states
    """
    np.random.seed(0)
    
    args = create_default_args(
        n=n,
        TopN=top_n,
        display_state=display,
        logging_mode=False
    )
    
    print(f"Running Advanced MCTS Demo: Grid Size {n}x{n}, Top-{top_n} Priority")
    evaluate_advanced_mcts(args)


def demo_batch_comparison():
    """
    Run a batch comparison across different configurations.
    """
    np.random.seed(0)
    
    n_list = np.arange(5, 55, 5)
    
    # Test Top-3 configuration
    print("=== Testing Top-3 Priority Configuration ===")
    base_args_top3 = create_default_args(
        C=0.2,
        num_searches=10_000,
        num_workers=4,
        virtual_loss=1.0,
        TopN=3
    )
    
    result_list_top3 = run_batch_evaluation(n_list, base_args_top3)
    
    # Test Top-1 configuration
    print("\n=== Testing Top-1 Priority Configuration ===")
    base_args_top1 = create_default_args(
        C=0.2,
        num_searches=10_000,
        num_workers=4,
        virtual_loss=1.0,
        TopN=1
    )
    
    result_list_top1 = run_batch_evaluation(n_list, base_args_top1)
    
    print("\n=== Results Summary ===")
    print("Top-3 Results:", result_list_top3)
    print("Top-1 Results:", result_list_top1)
    
    return result_list_top3, result_list_top1


if __name__ == "__main__":
    # Run demonstration
    demo_single_evaluation(n=10, top_n=3, display=True)
