# tests/env_test.py

def test_environment_installation():
    print("🔍 Importing libraries...")
    import torch
    import gymnasium
    import matplotlib.pyplot as plt
    import numpy as np

    print("✅ Environment test completed successfully.")

if __name__ == "__main__":
    try:
        test_environment_installation()
    except Exception as e:
        print("❌ Test failed due to the following error:")
        print(e)
