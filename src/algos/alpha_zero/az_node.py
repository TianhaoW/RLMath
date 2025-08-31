import math
import numpy as np
import torch
import threading
import concurrent.futures # For parallel expansion
from .helper_functions.az_exploration_decay import exploration_decay_nb
from .helper_functions.bit_pack_util import _pack_bits_bool2d, _unpack_bits_to_2d

class Node_AZ:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0.0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = float(prior)

        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)

        if parent is None:
            self.level = int(np.sum(state))  # Level is the number of points placed
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_with_symmetry')):
                self.action_space = game.get_valid_moves(state)
                self.valid_moves = game.filter_valid_moves_by_symmetry(
                    self.action_space, state
                ).copy()
            else:
                self.valid_moves = game.get_valid_moves(state)
                self.action_space = self.valid_moves.copy()
        else:
            self.level = parent.level + 1
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_subset_with_symmetry')):
                self.action_space = game.get_valid_moves_subset(
                    parent.state, parent.action_space, self.action_taken)
                self.valid_moves = game.filter_valid_moves_by_symmetry(
                    self.action_space, state
                ).copy()
            else:
                self.valid_moves = game.get_valid_moves_subset(
                    parent.state, parent.action_space, self.action_taken)
                self.action_space = self.valid_moves.copy()
        
        # Ensure action_space is immutable
        self.action_space.flags.writeable = False

        self.is_full = False
        self._cached_ucb = None     # Cached UCB value
        self._ucb_dirty = True      # Indicates whether the cached UCB is stale
    
    # Keep for compatibility, but not used in single thread AlphaZero
    def apply_virtual_loss(self):
        with self.lock:
            self.value_sum -= self._vl
            self.visit_count += 1
            self._ucb_dirty = True

    def revert_virtual_loss(self):
        with self.lock:
            self.value_sum += self._vl
            self._ucb_dirty = True

    def is_fully_expanded(self):
        return self.is_full

    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    # PUCT formula
    def get_ucb(self, child, iter):
        parent_visit = max(1, self.visit_count)
        q = child.q_value()
        if self.args['exploration_decay'] and iter is not None:
            c = self.args['C'] * exploration_decay_nb(iter/self.args['num_searches'])
        else:
            c = self.args['C']
        # Consider try different exloration functions
        u = c * child.prior * math.sqrt(math.log(parent_visit)) / (1 + child.visit_count)
        return q + u
    
    def select(self, iter=None):
        best_child = None
        best_score = -1e18
        for child in self.children:
            score = self.get_ucb(child, iter)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    # New expand function using policy vector
    def expand_with_policy(self, policy_vec: np.ndarray):

        if self.is_full:
            return
        
        for action, prob in enumerate(policy_vec):
            if prob > 0 and self.valid_moves[action] == 1:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)
                child = Node_AZ(self.game, self.args, child_state, self, action, prior=prob)
                self.children.append(child)

        self.is_full = True

        if len(self.children) == 0:
            self.is_full = True


    def backpropagate(self, value):
        with self.lock:
            self.value_sum += value
            self._ucb_dirty = True  # Mark UCB as outdated
            self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

class Node_Compressed_AZ:
    """
    AlphaZero compatible Node that implements bit-packing for memory efficiency.
    - Stores state / valid_moves / action_space as bit-packed arrays.
    - Provides properties (.state, .valid_moves) to return unpacked views for compatibility.
    - Aligned with Node_AZ for use in an AlphaZero MCTS search.
    """
    __slots__ = (
        'game', 'args', 'parent', 'action_taken', 'prior',
        'children', 'visit_count', 'value_sum', 'lock', '_vl',
        'level', 'is_full',
        # packed payloads
        '_rows', '_cols', '_state_bits', '_valid_bits', '_action_bits'
    )

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0.0, visit_count=0):
        self.game = game
        self.args = args
        self.parent = parent
        self.action_taken = action_taken
        self.prior = float(prior)

        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0.0
        self.lock = threading.Lock()
        self._vl = args.get('virtual_loss', 1.0)

        self._rows = getattr(game, 'row_count', state.shape[0])
        self._cols = getattr(game, 'column_count', state.shape[1] if state.ndim > 1 else self._rows)

        self._state_bits = _pack_bits_bool2d(state)

        if parent is None:
            self.level = int(np.sum(state))
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_with_symmetry')):
                action_space = game.get_valid_moves(state)
                valid_moves  = game.filter_valid_moves_by_symmetry(action_space, state).copy()
            else:
                valid_moves  = game.get_valid_moves(state)
                action_space = valid_moves.copy()
        else:
            self.level = parent.level + 1
            parent_state = parent.state
            parent_action_space = parent.action_space
            if (self.level <= game.max_level_to_use_symmetry and 
                hasattr(game, 'get_valid_moves_subset_with_symmetry')):
                action_space = game.get_valid_moves_subset(parent_state, parent_action_space, self.action_taken)
                valid_moves  = game.filter_valid_moves_by_symmetry(action_space, self.state).copy()
            else:
                valid_moves  = game.get_valid_moves_subset(parent_state, parent_action_space, self.action_taken)
                action_space = valid_moves.copy()

        self._valid_bits  = _pack_bits_bool2d(valid_moves.reshape(self._rows, self._cols))
        self._action_bits = _pack_bits_bool2d(action_space.reshape(self._rows, self._cols))

        self.is_full = False

    @property
    def state(self) -> np.ndarray:
        return _unpack_bits_to_2d(self._state_bits, self._rows, self._cols)

    @property
    def valid_moves(self) -> np.ndarray:
        vm = _unpack_bits_to_2d(self._valid_bits, self._rows, self._cols).reshape(-1)
        vm.flags.writeable = False
        return vm

    @property
    def action_space(self) -> np.ndarray:
        am = _unpack_bits_to_2d(self._action_bits, self._rows, self._cols).reshape(-1)
        am.flags.writeable = False
        return am

    def apply_virtual_loss(self):
        with self.lock:
            self.value_sum -= self._vl
            self.visit_count += 1

    def revert_virtual_loss(self):
        with self.lock:
            self.value_sum += self._vl

    def is_fully_expanded(self):
        return self.is_full

    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def get_ucb(self, child, iter):
        parent_visit = max(1, self.visit_count)
        q = child.q_value()
        
        if self.args.get('exploration_decay', False):
            c = self.args['C'] * exploration_decay_nb(iter / self.args['num_searches'])
        else:
            c = self.args['C']
            
        u = c * child.prior * math.sqrt(math.log(parent_visit)) / (1 + child.visit_count)
        return q + u

    def select(self, iter):
        best_child = None
        best_score = -1e18
        for child in self.children:
            score = self.get_ucb(child, iter)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand_with_policy(self, policy_vec: np.ndarray):
        if self.is_full:
            return

        def _create_child(action_prob_tuple):
            action, prob = action_prob_tuple
            child_state = self.state.copy()
            child_state = self.game.get_next_state(child_state, action)
            return Node_Compressed_AZ(self.game, self.args, child_state, self, action, prior=prob)

        tasks = []
        current_valid_moves = self.valid_moves
        for action, prob in enumerate(policy_vec):
            if prob > 0 and current_valid_moves[action] == 1:
                tasks.append((action, prob))

        num_workers = self.args.get('num_workers', 1)
        if num_workers > 1 and len(tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                self.children = list(executor.map(_create_child, tasks))
        else:
            self.children = [_create_child(task) for task in tasks]

        self.is_full = True

    def backpropagate(self, value):
        node = self
        while node is not None:
            with node.lock:
                node.value_sum += value
                node.visit_count += 1
            node = node.parent

