# AlphaZero training loop for single-player N3il optimization
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from .az_mcts import MCTS_AZ

class AlphaZero:
    """
    AlphaZero with optional post-batch GPU data augmentation (D4).
    Set args:
        data_augmentation_per_traj=False            # disable per-episode CPU aug
        post_batch_augmentation=True       # enable one-shot GPU aug after self-play
        keep_aug_on_device=True            # keep augmented tensors on device (no CPU round trip)
    """
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS_AZ(game, args, model)

        seed = args.get('random_seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.policy_loss_coef = args.get('policy_loss_coef', 1.0)
        self.value_loss_coef  = args.get('value_loss_coef', 1.0)
        self.gradient_clip    = args.get('gradient_clip', None)

        # CPU numpy symmetry funcs (legacy / fallback per-trajectory)
        self._symmetry_funcs = [
            ("id",        lambda x: x),
            ("rot90",     lambda x: np.rot90(x, 1)),
            ("rot180",    lambda x: np.rot90(x, 2)),
            ("rot270",    lambda x: np.rot90(x, 3)),
            ("flip_h",    lambda x: np.flipud(x)),
            ("flip_v",    lambda x: np.fliplr(x)),
            ("flip_diag", lambda x: x.T),
            ("flip_anti", lambda x: np.rot90(x.T, 2)),
        ]
        # Torch batch transforms (operate on (B,C,n,n))
        self._torch_tf = [
            ("id",        lambda t: t),
            ("rot90",     lambda t: torch.rot90(t, 1, (-2, -1))),
            ("rot180",    lambda t: torch.rot90(t, 2, (-2, -1))),
            ("rot270",    lambda t: torch.rot90(t, 3, (-2, -1))),
            ("flip_h",    lambda t: torch.flip(t, (-2,))),         # vertical axis in image => flip rows
            ("flip_v",    lambda t: torch.flip(t, (-1,))),         # horizontal axis => flip cols
            ("flip_diag", lambda t: t.transpose(-2, -1)),
            ("flip_anti", lambda t: torch.rot90(t.transpose(-2, -1), 2, (-2, -1))),
        ]
        self._policy_perms = None  # (8, A) long tensor of flattened index permutations

    def _apply_temperature(self, action_probs, step_index):
        temp = self.args.get('temperature', 1.0)
        schedule = self.args.get('temperature_schedule')
        if callable(schedule):
            temp = schedule(step_index)
        if temp <= 0:
            greedy = np.zeros_like(action_probs)
            greedy[np.argmax(action_probs)] = 1.0
            return greedy
        p = np.power(action_probs, 1.0 / temp)
        s = p.sum()
        if s <= 0:
            return np.ones_like(p) / len(p)
        return p / s

    # Per-trajectory CPU augmentation (kept for backward compatibility)
    def _augment_trajectory(self, trajectory):
        augmented = []
        for encoded, policy_vec, value in trajectory:
            C, n, _ = encoded.shape
            policy_2d = policy_vec.reshape(n, n)
            for name, tf in self._symmetry_funcs:
                enc_tf = np.stack([tf(encoded[c]) for c in range(C)], axis=0)
                pol_tf = tf(policy_2d).reshape(-1)
                augmented.append((enc_tf, pol_tf, value))
        return augmented

    def _build_policy_perms(self, n, device):
        base = np.arange(n * n).reshape(n, n)
        perms = []
        for name, tf in self._symmetry_funcs:
            perms.append(tf(base).reshape(-1))
        perm_np = np.stack(perms, axis=0)  # (8, A)
        self._policy_perms = torch.as_tensor(perm_np, dtype=torch.long, device=device)

    def _augment_memory_batch(self, memory):
        """
        GPU batch augmentation.
        memory: list of (encoded_state(C,n,n), policy_vec(A), value)
        Returns list with 8x entries. States/policies may remain as torch.Tensor if keep_aug_on_device.
        """
        if not memory:
            return memory
        device = self.model.device
        states = torch.tensor(
            np.stack([s for s, _, _ in memory], axis=0),
            dtype=torch.float32, device=device
        )  # (B,C,n,n)
        policies = torch.tensor(
            np.stack([p for _, p, _ in memory], axis=0),
            dtype=torch.float32, device=device
        )  # (B,A)
        values = torch.tensor(
            [v for _, _, v in memory],
            dtype=torch.float32, device=device
        )  # (B,)

        B, C, n, _ = states.shape
        A = n * n
        if self._policy_perms is None or self._policy_perms.shape[1] != A:
            self._build_policy_perms(n, device)

        aug_states = []
        aug_pols = []
        for idx, (name, tf) in enumerate(self._torch_tf):
            st = tf(states)
            perm = self._policy_perms[idx]
            pol = policies.index_select(1, perm)
            aug_states.append(st)
            aug_pols.append(pol)

        aug_states = torch.stack(aug_states, 1).reshape(B * 8, C, n, n)
        aug_pols   = torch.stack(aug_pols, 1).reshape(B * 8, A)
        aug_vals   = values.repeat_interleave(8)

        if self.args.get('keep_aug_on_device', True):
            return [(aug_states[i], aug_pols[i], aug_vals[i].item()) for i in range(aug_states.size(0))]

        # Move back to CPU numpy if requested
        aug_states_np = aug_states.cpu().numpy()
        aug_pols_np   = aug_pols.cpu().numpy()
        aug_vals_np   = aug_vals.cpu().numpy()
        return [(aug_states_np[i], aug_pols_np[i], aug_vals_np[i]) for i in range(aug_states_np.shape[0])]

    def self_play(self):
        """
        Run one full episode.
        Returns list of (encoded_state, policy_target, value_target).
        (No augmentation here if using post_batch_augmentation.)
        """
        memory = []
        state = self.game.get_initial_state()
        step = 0
        while True:
            action_probs = self.mcts.search(state)
            stored_state = state.copy()
            memory.append((stored_state, action_probs))
            tau_policy = self._apply_temperature(action_probs, step)
            action = np.random.choice(self.game.action_size, p=tau_policy)
            state = self.game.get_next_state(state.copy(), action)
            valid_moves = self.game.get_valid_moves(state)
            value, terminal = self.game.get_value_and_terminated(state, valid_moves)
            if terminal:
                trajectory = []
                for hist_state, hist_policy in memory:
                    encoded = self.game.get_encoded_state(hist_state)
                    trajectory.append((encoded, hist_policy, value))
                # Per-trajectory CPU augmentation only if enabled and not using batch aug
                if self.args.get('data_augmentation_per_traj', False) and not self.args.get('post_batch_augmentation', False):
                    trajectory = self._augment_trajectory(trajectory)
                return trajectory
            step += 1

    def _policy_value_loss(self, logits, values, policy_targets, value_targets):
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)
        loss = self.policy_loss_coef * policy_loss + self.value_loss_coef * value_loss
        return loss, policy_loss.item(), value_loss.item()

    def train_epoch(self, memory, epoch_idx=None):
        random.shuffle(memory)
        batch_size = self.args.get('batch_size', 32)
        show_batches = self.args.get('show_train_progress', True)
        self.model.train()
        metrics = []

        iterator = range(0, len(memory), batch_size)
        if show_batches:
            iterator = tqdm(iterator,
                            desc=f"Epoch {epoch_idx} Batches",
                            leave=False)

        running_loss = running_pl = running_vl = 0.0

        for bi, start in enumerate(iterator):
            batch = memory[start:start + batch_size]
            if not batch:
                continue
            states, policy_targets, value_targets = zip(*batch)

            if isinstance(states[0], torch.Tensor):
                states_t = torch.stack(states, 0).to(self.model.device)
            else:
                states_t = torch.tensor(np.array(states),
                                        dtype=torch.float32,
                                        device=self.model.device)

            if isinstance(policy_targets[0], torch.Tensor):
                policy_t = torch.stack(policy_targets, 0).to(self.model.device)
            else:
                policy_t = torch.tensor(np.array(policy_targets),
                                        dtype=torch.float32,
                                        device=self.model.device)

            value_t = torch.tensor(np.array(value_targets).reshape(-1, 1),
                                   dtype=torch.float32,
                                   device=self.model.device)

            logits, pred_values = self.model(states_t)
            loss, pl, vl = self._policy_value_loss(logits, pred_values, policy_t, value_t)
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            metrics.append((loss.item(), pl, vl))
            running_loss += loss.item()
            running_pl += pl
            running_vl += vl

            if show_batches:
                avg_loss = running_loss / (bi + 1)
                avg_pl   = running_pl / (bi + 1)
                avg_vl   = running_vl / (bi + 1)
                iterator.set_postfix(loss=f"{avg_loss:.3f}",
                                     pl=f"{avg_pl:.3f}",
                                     vl=f"{avg_vl:.3f}")

        if metrics:
            return {
                'loss': np.mean([m[0] for m in metrics]),
                'policy_loss': np.mean([m[1] for m in metrics]),
                'value_loss': np.mean([m[2] for m in metrics]),
            }
        return {}

    def learn(self):
        num_iterations = self.args.get('num_iterations', 1)
        num_selfplay   = self.args.get('num_selfPlay_iterations', 1)
        num_epochs     = self.args.get('num_epochs', 1)
        save_interval  = self.args.get('save_interval', 1)
        weight_prefix  = self.args.get('weight_file_name', 'alphazero')

        import os
        weights_dir = self.args.get('weights_dir', None)
        if weights_dir is not None:
            os.makedirs(weights_dir, exist_ok=True)

        all_stats = []
        for it in range(num_iterations):
            print(f"=== AlphaZero Iteration {it}/{num_iterations} ===")
            self.model.eval()
            memory = []
            for i in trange(num_selfplay, desc=f"SelfPlay Iter: ", leave=False):
                print(f" Self-play episode {i}/{num_selfplay}")
                memory.extend(self.self_play())

            if self.args.get('post_batch_augmentation', False):
                print(" Applying post-self-play data augmentation on GPU...")
                memory = self._augment_memory_batch(memory)
            
            print(f" Training on {len(memory)} samples...")
            epoch_stats = []
            for ep in trange(num_epochs, desc="Train Epochs", leave=True):
                stats = self.train_epoch(memory, epoch_idx=ep)
                epoch_stats.append(stats)
            all_stats.append(epoch_stats)

            if (it + 1) % save_interval == 0:
                if weights_dir is None:
                    model_path = f"{weight_prefix}_model_{it+1}.pt"
                    opt_path   = f"{weight_prefix}_optimizer_{it+1}.pt"
                else:
                    model_path = os.path.join(weights_dir, f"{weight_prefix}_model_{it+1}.pt")
                    opt_path   = os.path.join(weights_dir, f"{weight_prefix}_optimizer_{it+1}.pt")
                torch.save(self.model.state_dict(), model_path)
                torch.save(self.optimizer.state_dict(), opt_path)
                if self.args.get('logging_mode', False):
                    print(f"Saved: {model_path}  {opt_path}")
        return all_stats

    @torch.no_grad()
    def inference_step(self, state, temperature: float = 0.0, return_net: bool = True):
        """
        Single-step inference helper.
        Args:
            state (np.ndarray): raw game state.
            temperature (float): sampling temperature (0 => argmax).
            return_net (bool): if True also return raw network policy/value.
        Returns:
            (action, mcts_policy[, net_policy, net_value])
        """
        old_pb = self.args.get('process_bar', False)
        old_eps = self.args.get('dirichlet_epsilon', 0.0)
        self.args['process_bar'] = False
        self.args['dirichlet_epsilon'] = 0.0

        mcts_policy = self.mcts.search(state)

        self.args['process_bar'] = old_pb
        self.args['dirichlet_epsilon'] = old_eps

        if temperature <= 0:
            pi = np.zeros_like(mcts_policy)
            pi[np.argmax(mcts_policy)] = 1.0
        else:
            p = np.power(mcts_policy, 1.0 / temperature)
            s = p.sum()
            pi = p / s if s > 0 else np.ones_like(p) / len(p)

        action = int(np.argmax(pi))

        if not return_net:
            return action, mcts_policy

        encoded = self.game.get_encoded_state(state)
        t = torch.tensor(encoded, dtype=torch.float32, device=self.model.device).unsqueeze(0)
        logits, value = self.model(t)
        net_policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
        net_value = float(value.item())
        return action, mcts_policy, net_policy, net_value