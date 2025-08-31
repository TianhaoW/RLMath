import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.algos.mcts import evaluate, MCTS
from src.envs.collinear_for_mcts import N3il
from src.algos.alpha_zero.alpha_zero import AlphaZero
from src.algos.alpha_zero.nn.az_ResNet import ResNet
import torch

if __name__ == "__main__":

    n = 3

    args = {
        'environment': 'N3il',
        'algorithm': 'AlphaZero',
        'max_level_to_use_symmetry': -1,
        'n': n,
        'C': 1.41,
        'exploration_decay': True,          # keep False unless using decay
        'dirichlet_epsilon': 0,           # >0 to add root noise
        'dirichlet_alpha': 0.3,
        'num_searches': 10*(n**2),                 # per move MCTS simulations
        'num_workers': 1,
        'virtual_loss': 1.0,
        'process_bar': True,
        'display_state': False,
        'logging_mode': True,
        'TopN': 2*n,
        'simulate_with_priority': False,
        'random_seed': 1,
        'tree_visualization': False,
        'node_compression': False,           # set True to use Node_Compressed_AZ
        # ---- AlphaZero training hyperparams ----
        'num_iterations': 5,                # outer loop
        'num_selfPlay_iterations': 5,       # episodes per iteration
        'num_epochs': 5,                     # train epochs per iteration
        'batch_size': 64,
        'temperature': 0.0,                  # sampling temperature
        # 'temperature_schedule': lambda step: 1.0 if step < 4 else 0.1,  # optional
        'policy_loss_coef': 1.0,
        'value_loss_coef': 1.0,
        'gradient_clip': 5.0,
        'save_interval': 1,
        # --- Data augmentation ---
        'data_augmentation': False,          # 
        'post_batch_augmentation': True,     # Data Aug in GPU after all self-play
        'keep_aug_on_device': True,          # keep on GPU
        # Optional evaluation / early stop hooks
        # 'early_stop_value_threshold': 0.98,
        # 'eval_interval': 5,
        # Paths
        'weight_file_name': f'az_n{n}',
        'weights_dir': os.path.join(os.path.dirname(__file__), 'weights'),
        'table_dir': os.path.dirname(__file__), 
        'figure_dir': os.path.join(os.path.dirname(__file__), 'figure'),
        'checkpoint_dir': os.path.join(os.path.dirname(__file__), 'checkpoints'),
    }

    # Rebuild environment (uses existing args)
    n = args['n']
    env = N3il((n, n), args)

    # Model, optimizer, (optionally scheduler)
    model = ResNet(env, num_resBlocks=4, num_hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    az = AlphaZero(model, optimizer, env, args)

    print("Start training...")
    train_stats = az.learn()
    print("Training finished.")