#!/usr/bin/env python3
"""
Quick Advanced MCTS Test - Fast verification of integration

This script provides a quick test of the advanced MCTS functionality
without running the full comprehensive evaluation.
"""

import os
import sys
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_test_basic_functionality():
    """Test basic functionality quickly."""
    print("QUICK ADVANCED MCTS TEST")
    print("=" * 40)
    
    try:
        from src.evaluation_advanced import create_default_args, evaluate_advanced_mcts
        
        print("✓ Advanced evaluation module imported successfully")
        
        # Test with very small grid and few simulations
        args = create_default_args(
            n=5,
            num_searches=100,  # Very small for speed
            display_state=False,
            logging_mode=True,
            process_bar=False,
            num_workers=1  # Single thread for simplicity
        )
        
        print("✓ Configuration created")
        
        # Run a single quick evaluation
        result = evaluate_advanced_mcts(args)
        
        print(f"✓ Advanced MCTS completed successfully!")
        print(f"  Result: Placed {result} points on 5x5 grid")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test_components():
    """Test individual components quickly."""
    print("\nTesting Individual Components:")
    print("-" * 30)
    
    try:
        # Test priority grid loading
        from src.priority_advanced import ensure_priority_grid_exists
        priority_grid = ensure_priority_grid_exists(5)
        print(f"✓ Priority grid loaded: shape {priority_grid.shape}")
        
        # Test advanced environment
        from src.algos.mcts_advanced import AdvancedN3ilEnvironment
        env = AdvancedN3ilEnvironment(
            grid_size=(5, 5), 
            args={'TopN': 2}, 
            priority_grid=priority_grid
        )
        print("✓ Advanced environment created")
        
        # Test state operations
        state = env.get_initial_state()
        valid_moves = env.get_valid_moves(state)
        print(f"✓ Valid moves calculated: {np.sum(valid_moves)} available")
        
        # Test MCTS
        from src.algos.mcts_advanced import AdvancedMCTS
        mcts = AdvancedMCTS(env, {'num_searches': 50, 'C': 1.4, 'process_bar': False})
        
        action_probs = mcts.search(state)
        print(f"✓ MCTS search completed: {np.sum(action_probs > 0)} actions with non-zero probability")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_comparison_test():
    """Quick comparison between different approaches."""
    print("\nQuick Performance Comparison:")
    print("-" * 30)
    
    try:
        # Test different TopN settings quickly
        from src.evaluation_advanced import create_default_args, evaluate_advanced_mcts
        
        configs = [
            ("Top-1", {'TopN': 1, 'num_searches': 50}),
            ("Top-3", {'TopN': 3, 'num_searches': 50}),
        ]
        
        results = {}
        
        for config_name, config_params in configs:
            args = create_default_args(
                n=5,
                display_state=False,
                logging_mode=True,
                process_bar=False,
                num_workers=1,
                **config_params
            )
            
            # Run one quick test
            result = evaluate_advanced_mcts(args)
            results[config_name] = result
            print(f"  {config_name}: {result} points")
        
        print("✓ Comparison completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False


def test_migration_guide():
    """Test the migration guide functionality."""
    print("\nTesting Migration Guide:")
    print("-" * 30)
    
    try:
        # Import and test the migration guide
        exec(open('luoning_migration_guide.py').read())
        print("✓ Migration guide executed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Migration guide test failed: {e}")
        return False


def main():
    """Run quick tests."""
    print("QUICK TEST SUITE FOR ADVANCED MCTS INTEGRATION")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", quick_test_basic_functionality),
        ("Individual Components", quick_test_components),
        ("Quick Comparison", quick_comparison_test),
        ("Migration Guide", test_migration_guide),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"QUICK TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Advanced MCTS integration is working correctly.")
        print("\nYou can now:")
        print("1. Run the migration guide: python luoning_migration_guide.py")
        print("2. Use the modernized notebook: luoning_modernized_mcts.ipynb")
        print("3. Experiment with different configurations")
        print("4. Integrate with your own research")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
