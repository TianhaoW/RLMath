import torch
import numpy as np
from src.registry.algo_registry import get_algo
from src.registry.env_registry import get_env
from src.config_utils import parse_config, get_logger
from src.envs import NoThreeCollinearEnvWithPriority, Point
from src.algos.mcts_unified import UnifiedMCTS, N3ilUnified, SupNormPriority, evaluate_unified

def test_mcts_algorithms():
    """Test all MCTS algorithm variants."""
    print("Testing MCTS Algorithms")
    print("=" * 50)
    
    config = {
        'n': 4,
        'num_searches': 100,
        'C': 1.414,
        'top_n': 2,
        'num_workers': 2,
        'virtual_loss': 1.0,
        'priority_type': 'supnorm',
        'display_state': False,
        'process_bar': False,
    }
    
    variants = ['basic', 'priority', 'parallel', 'advanced']
    results = {}
    
    for variant in variants:
        print(f"\nTesting {variant} MCTS...")
        try:
            # Test registry access
            mcts_constructor = get_algo(f'mcts_{variant}')
            mcts = mcts_constructor(config)
            
            # Test direct MCTS evaluation instead of evaluate_unified
            unified_env = N3ilUnified(
                grid_size=(config['n'], config['n']), 
                args=config, 
                priority_system=SupNormPriority()
            )
            mcts_direct = UnifiedMCTS(unified_env, config, variant=variant)
            
            # Run a quick game
            game_state = unified_env.get_initial_state()
            moves = 0
            while moves < 5:  # Just test a few moves
                valid_moves = unified_env.get_valid_moves(game_state)
                value, is_terminal = unified_env.get_value_and_terminated(game_state, valid_moves)
                if is_terminal:
                    break
                action_probs = mcts_direct.search(game_state.copy())
                action = np.argmax(action_probs)
                game_state = unified_env.get_next_state(game_state, action)
                moves += 1
            
            points = np.sum(game_state)
            results[variant] = points
            
            print(f"  ✓ {variant} MCTS: {points} points in {moves} moves")
            
        except Exception as e:
            print(f"  ✗ {variant} MCTS failed: {e}")
            results[variant] = None
    
    print(f"\nMCTS Test Results:")
    print("-" * 30)
    for variant, points in results.items():
        status = "✓" if points is not None else "✗"
        score = f"{points} points" if points is not None else "FAILED"
        print(f"  {status} {variant:8}: {score}")
    
    return results

def test_dqn_model(model_path):
    """Test a trained DQN model if it exists."""
    print(f"\nTesting DQN Model: {model_path}")
    print("=" * 50)
    
    try:
        # Check if model file exists
        import os
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        # Load model and test
        # This would require the actual DQN model loading logic
        print(f"Model file found: {model_path}")
        print("DQN model testing not implemented yet")
        return None
        
    except Exception as e:
        print(f"Error testing DQN model: {e}")
        return None

def test_environment():
    """Test environment functionality."""
    print("\nTesting Environment")
    print("=" * 50)
    
    try:
        # Test priority function
        def priority_function(p: Point, grid_size) -> float:
            x, y = p.x, p.y
            m, n = grid_size
            center_x, center_y = (m - 1) / 2, (n - 1) / 2
            return max(abs(x - center_x), abs(y - center_y)) / max(center_x, center_y)
        
        # Create environment
        env = NoThreeCollinearEnvWithPriority(4, 4, priority_function)
        env.reset()
        
        # Test basic operations
        point = Point(0, 0)
        env.self_play_add_point(point)
        
        print("  ✓ Environment creation successful")
        print("  ✓ Point addition successful")
        
        # Test if plot method exists
        if hasattr(env, 'plot'):
            print("  ✓ Environment has plot() method")
        else:
            print("  ✗ Environment missing plot() method")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("RLMath Model Testing Suite")
    print("=" * 60)
    
    # Test environment
    env_success = test_environment()
    
    # Test MCTS algorithms
    mcts_results = test_mcts_algorithms()
    
    # Test DQN model if specified
    config = parse_config()
    env_cfg = config["env"]
    m, n = env_cfg['m'], env_cfg['n']
    model_name = env_cfg.get('model', 'mcts')
    env_name = env_cfg['env_type']
    algo_name = config["algo"]["method"]
    
    if not algo_name.startswith('mcts'):
        model_file = config['path']['model_dir'] / f"{env_name}_{algo_name}_{model_name}_{m}x{n}.pt"
        test_dqn_model(model_file)
    
    # Summary
    print(f"\nTest Summary")
    print("=" * 60)
    print(f"Environment: {'✓ PASS' if env_success else '✗ FAIL'}")
    
    mcts_success = sum(1 for r in mcts_results.values() if r is not None)
    print(f"MCTS Algorithms: {mcts_success}/{len(mcts_results)} variants working")
    
    if mcts_success == len(mcts_results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above.")

if __name__ == "__main__":
    main()
