from tqdm import trange
import torch
import numpy as np
from .az_node import Node_AZ, Node_Compressed_AZ
from .helper_functions.mcts_visualizer import MCTSVisualizer

class MCTS_AZ:
    def __init__(self, game, args, model):
        self.game = game
        self.model = model
        self.args = args
        self.trial_id = None # Current trial ID

        # If visualization is enabled, create a visualizer instance
        if self.args.get('tree_visualization', False):
            self.visualizer = MCTSVisualizer(self.game, self.args)
        else:
            self.visualizer = None

    def start_new_trial(self, trial_id):
        """Starts a new trial, like a new game or experiment run."""
        self.trial_id = trial_id
        if self.visualizer:
            self.visualizer.start_new_trial(trial_id)

    @torch.no_grad()
    def _infer(self, state):
        encoded_state = self.game.get_encoded_state(state)  # Game must provide encoding
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.model.device).unsqueeze(0)
        logits, value = self.model(state_tensor)
        policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return policy, float(value.item())

    @torch.no_grad()
    def search(self, state):
        # define root
        if self.args.get('node_compression', False):
            root = Node_Compressed_AZ(self.game, self.args, state)
            print("Using Node_Compressed_AZ for MCTS")
        else:
            root = Node_AZ(self.game, self.args, state)

        # Root inference
        policy, value = self._infer(state)
        
        # Masking
        policy *= root.valid_moves
        policy_sum = policy.sum()
        if policy_sum <= 0:
            # Uniform over legal moves if policy is all zero
            valid_moves = root.valid_moves
            valid_moves_count = valid_moves.sum()
            if valid_moves_count > 0:
                policy = valid_moves / valid_moves_count
        else:
            policy /= policy_sum

        # Add Dirichlet noise for exploration at the root
        eps = self.args.get('dirichlet_epsilon', 0.0)
        if eps > 0:
            alpha = self.args.get('dirichlet_alpha', 0.3)
            noise = np.random.dirichlet([alpha] * self.game.action_size)
            policy = (1 - eps) * policy + eps * noise
            # Re-mask and normalize
            policy *= root.valid_moves
            policy_sum_after_noise = policy.sum()
            if policy_sum_after_noise > 0:
                policy /= policy_sum_after_noise

        root.expand_with_policy(policy)
        root.backpropagate(value)

        if self.args.get('process_bar', True):
            search_iterator = trange(self.args['num_searches'])
        else:
            search_iterator = range(self.args['num_searches'])

        for i in search_iterator:
            node = root
            # Selection
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.select(iter=i)

            # Terminal state check
            term_value, terminal = self.game.get_value_and_terminated(node.state, node.valid_moves)
            if terminal:
                node.backpropagate(term_value)
                continue

            # Infer and expand
            policy, value = self._infer(node.state)
            policy *= node.valid_moves
            policy_sum = policy.sum()
            if policy_sum <= 0:
                valid_moves = node.valid_moves
                valid_moves_count = valid_moves.sum()
                if valid_moves_count > 0:
                    policy = valid_moves / valid_moves_count
            else:
                policy /= policy_sum
            
            node.expand_with_policy(policy)
            node.backpropagate(value)

        # Calculate action probabilities based on visit counts
        action_probs = np.zeros(self.game.action_size, dtype=np.float32)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        
        total_visits = action_probs.sum()
        if total_visits > 0:
            action_probs /= total_visits
        
        # ---- DELEGATE VISUALIZATION ----
        if self.visualizer:
            num_points = np.sum(state)
            snapshot_name = f"Step {num_points}: {num_points} points placed"
            
            # Call the visualizer to create the snapshot
            self.visualizer.create_tree_snapshot(root, snapshot_name)
            
            if self.args.get('pause_at_each_step', False):
                try:
                    response = input("Output action prob? (y/n): ").strip().lower()
                    if response == 'y':
                        print("Action probabilities:", action_probs)
                except (EOFError, KeyboardInterrupt):
                    pass
        
        return action_probs