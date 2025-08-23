"""Memory comparison test between regular Node and Node_Compressed.

This test performs exactly ONE MCTS .search() style run (i.e. a batch of iterations
from the empty grid) using:
  1) Regular Node storage
  2) Compressed Node storage (bit-packed)

It measures resident memory (RSS) before and after the search, reports deltas,
and prints a per-node average if possible.

The search loop mirrors the provided reference snippet (selection -> expansion -> simulation -> backprop).
Only this file is modified per user request.
"""

from __future__ import annotations

import os
import sys
import time
import gc
import psutil
import numpy as np
from typing import Dict, Any, Tuple, List, Set
import tracemalloc
import sys as _sys

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tqdm import trange

try:
    from src.algos.mcts import MCTS, Node, Node_Compressed
except ImportError:
    from src.algos.mcts import MCTS, Node  # Fallback if Node_Compressed missing
    Node_Compressed = None  # type: ignore

from src.envs import N3il, N3il_with_symmetry


PROC = psutil.Process(os.getpid())

def rss_mb() -> float:
    return PROC.memory_info().rss / 1024 / 1024

def format_mb(x: float) -> str:
    return f"{x:.3f} MB"


def count_nodes(root) -> int:
    if root is None:
        return 0
    total = 1
    for c in getattr(root, "children", []) or []:
        total += count_nodes(c)
    return total


def single_search(game, args: Dict[str, Any], sample_rss: bool=False, sample_every: int=1000):
    """Run one MCTS search loop (args['num_searches'] iterations).

    Returns:
        (action_probs, root, rss_samples)
        rss_samples: list of (iteration, rss_mb) if sampling enabled else []
    """
    # Define root (compressed or regular)
    if args.get('node_compression', False) and Node_Compressed is not None:
        root = Node_Compressed(game, args, game.get_initial_state().copy())
        print("Using Node_Compressed for MCTS")
    else:
        root = Node(game, args, game.get_initial_state().copy())

    # Choose iterator
    if args.get('process_bar', False):
        iterator = trange(args['num_searches'], desc='MCTS')
    else:
        iterator = range(args['num_searches'])

    rss_samples: List[Tuple[int, float]] = []
    for search in iterator:
        node = root

        # Selection: descend while fully expanded
        while node.is_fully_expanded():  # expects is_full and has children
            node = node.select(iter=search)

        # Evaluate & expand
        if node.action_taken is not None:
            value, is_terminal = game.get_value_and_terminated(node.state, node.valid_moves)
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
        else:
            node = node.expand()
            value = node.simulate()

        # Backpropagate
        node.backpropagate(value)

    # Build action probabilities
    action_probs = np.zeros(game.action_size, dtype=np.float32)
    total_visits = 0
    for child in root.children:
        v = child.visit_count
        action_probs[child.action_taken] = v
        total_visits += v
    if total_visits > 0:
        action_probs /= total_visits

        if sample_rss and (search % sample_every == 0 or search+1 == args['num_searches']):
            rss_samples.append((search, rss_mb()))

    return action_probs, root, rss_samples


def warmup_numba(game_factory, n: int):
    """Do a small dummy search to trigger JIT before measurement."""
    small_args = {
        'environment': 'N3il_with_symmetry',
        'algorithm': 'MCTS',
        'node_compression': False,
        'max_level_to_use_symmetry': 1,
        'n': n,
        'C': 1.41,
        'num_searches': 50,
        'num_workers': 1,
        'virtual_loss': 1.0,
        'process_bar': True,
        'display_state': False,
        'logging_mode': False,
        'TopN': n,
        'simulate_with_priority': False,
        'table_dir': os.path.dirname(__file__),
        'figure_dir': os.path.join(os.path.dirname(__file__), 'figure'),
        'random_seed': 123,
        'tree_visualization': False,
        'pause_at_each_step': False,
    }
    game = game_factory(small_args)
    single_search(game, small_args)


def aggregate_tree_memory(root) -> Dict[str, float]:
    """Approximate memory for the tree by traversing nodes and summing object + array sizes.

    Notes: sys.getsizeof covers Python object headers; numpy buffers via nbytes.
           Avoid double counting shared arrays using seen id set.
    Returns dict with totals in bytes.
    """
    totals = {
        'node_object': 0.0,
        'numpy_arrays': 0.0,
        'state_arrays': 0.0,
        'bit_arrays': 0.0,
        'total': 0.0,
        'node_count': 0,
    }
    seen: Set[int] = set()

    stack = [root]
    while stack:
        node = stack.pop()
        totals['node_count'] += 1
        try:
            totals['node_object'] += _sys.getsizeof(node)
        except Exception:
            pass

        # State (unpacked) if present
        if hasattr(node, 'state'):
            st = node.state
            sid = id(st)
            if isinstance(st, np.ndarray) and sid not in seen:
                seen.add(sid)
                totals['numpy_arrays'] += _sys.getsizeof(st)
                totals['state_arrays'] += st.nbytes

        # Valid moves array
        if hasattr(node, 'valid_moves'):
            vm = node.valid_moves
            vid = id(vm)
            if isinstance(vm, np.ndarray) and vid not in seen:
                seen.add(vid)
                totals['numpy_arrays'] += _sys.getsizeof(vm)
                totals['state_arrays'] += vm.nbytes

        # Compressed internals
        for name in ('_state_bits', '_valid_bits', '_action_bits'):
            if hasattr(node, name):
                arr = getattr(node, name)
                if isinstance(arr, np.ndarray):
                    aid = id(arr)
                    if aid not in seen:
                        seen.add(aid)
                        totals['numpy_arrays'] += _sys.getsizeof(arr)
                        totals['bit_arrays'] += arr.nbytes

        # Children
        for c in getattr(node, 'children', []) or []:
            stack.append(c)

    totals['total'] = totals['node_object'] + totals['numpy_arrays'] + totals['state_arrays'] + totals['bit_arrays']
    return totals


def print_tree_memory_stats(label: str, stats: Dict[str, float]):
    if stats['node_count'] == 0:
        print(f"{label}: empty tree")
        return
    nb = 1024.0
    per_node = (stats['state_arrays'] + stats['bit_arrays']) / stats['node_count'] if stats['node_count'] else 0
    print(f"{label} Node Count: {stats['node_count']}")
    print(f"  Node objects:       {stats['node_object']/nb/nb:.3f} MB")
    print(f"  NumPy overhead:     {stats['numpy_arrays']/nb/nb:.3f} MB (python object wrappers)")
    print(f"  Unpacked arrays:    {stats['state_arrays']/nb/nb:.3f} MB")
    print(f"  Bit-packed arrays:  {stats['bit_arrays']/nb/nb:.3f} MB")
    print(f"  Approx data/node:   {per_node/1024:.3f} KB (unpacked+bit)")


def run_memory_comparison():
    n = 40  # board size (increase for clearer diffs)
    seed = 0

    base_args = {
        'environment': 'N3il_with_symmetry',
        'algorithm': 'MCTS',
        'node_compression': False,  # toggled later
        'max_level_to_use_symmetry': 1,
        'n': n,
        'C': 1.41,
        'num_searches': 100 * (n ** 2),  # as in user snippet
        'num_workers': 1,
        'virtual_loss': 1.0,
        'process_bar': True,          # disable for clean test output
        'display_state': False,
        'logging_mode': False,
        'TopN': n,
        'simulate_with_priority': False,
        'table_dir': os.path.dirname(__file__),
        'figure_dir': os.path.join(os.path.dirname(__file__), 'figure'),
        'random_seed': seed,
        'tree_visualization': False,
        'pause_at_each_step': False,
    }

    def game_factory(local_args):
        if local_args['environment'] == 'N3il_with_symmetry':
            return N3il_with_symmetry(grid_size=(n, n), args=local_args, priority_grid=None)
        return N3il(grid_size=(n, n), args=local_args, priority_grid=None)

    # Warmup (JIT compile) once
    print("[Warmup JIT]")
    warmup_numba(game_factory, n)
    gc.collect(); time.sleep(0.05)

    print("=== Regular Node Run ===")
    base_args['node_compression'] = False
    game = game_factory(base_args)
    gc.collect(); time.sleep(0.05)
    tracemalloc.start()
    before_reg_rss = rss_mb()
    start_time = time.time()
    _, root_reg, samples_reg = single_search(game, base_args, sample_rss=True, sample_every=max(1000, base_args['num_searches']//20))
    t_reg = time.time() - start_time
    snap_reg = tracemalloc.take_snapshot()
    tracemalloc.stop()
    after_reg_rss = rss_mb()
    nodes_reg = count_nodes(root_reg)
    peak_reg_rss = max([s for _, s in samples_reg] + [after_reg_rss]) if samples_reg else after_reg_rss
    delta_reg_final = after_reg_rss - before_reg_rss
    delta_reg_peak = peak_reg_rss - before_reg_rss
    tree_stats_reg = aggregate_tree_memory(root_reg)
    print(f"Regular: nodes={nodes_reg}, time={t_reg:.3f}s, FinalΔRSS={delta_reg_final:.3f} MB, PeakΔRSS={delta_reg_peak:.3f} MB")
    print_tree_memory_stats("Regular Tree", tree_stats_reg)

    # Tracemalloc summary (top 3)
    top_stats = snap_reg.statistics('filename')[:3]
    print("  Tracemalloc top allocations (regular):")
    for st in top_stats:
        print(f"    {st.count} blocks: {st.size/1024/1024:.3f} MB -> {os.path.basename(st.traceback[0].filename)}")

    if Node_Compressed is None:
        print("Node_Compressed not available; skipping compressed run.")
        return

    print("\n=== Compressed Node Run ===")
    comp_args = dict(base_args)
    comp_args['node_compression'] = True
    game_c = game_factory(comp_args)
    gc.collect(); time.sleep(0.05)
    tracemalloc.start()
    before_comp_rss = rss_mb()
    start_time = time.time()
    _, root_comp, samples_comp = single_search(game_c, comp_args, sample_rss=True, sample_every=max(1000, comp_args['num_searches']//20))
    t_comp = time.time() - start_time
    snap_comp = tracemalloc.take_snapshot()
    tracemalloc.stop()
    after_comp_rss = rss_mb()
    nodes_comp = count_nodes(root_comp)
    peak_comp_rss = max([s for _, s in samples_comp] + [after_comp_rss]) if samples_comp else after_comp_rss
    delta_comp_final = after_comp_rss - before_comp_rss
    delta_comp_peak = peak_comp_rss - before_comp_rss
    tree_stats_comp = aggregate_tree_memory(root_comp)
    print(f"Compressed: nodes={nodes_comp}, time={t_comp:.3f}s, FinalΔRSS={delta_comp_final:.3f} MB, PeakΔRSS={delta_comp_peak:.3f} MB")
    print_tree_memory_stats("Compressed Tree", tree_stats_comp)
    top_stats_c = snap_comp.statistics('filename')[:3]
    print("  Tracemalloc top allocations (compressed):")
    for st in top_stats_c:
        print(f"    {st.count} blocks: {st.size/1024/1024:.3f} MB -> {os.path.basename(st.traceback[0].filename)}")

    # Summary
    print("\n=== Comparison Summary ===")
    # RSS comparison
    print(f"FinalΔ Regular vs Compressed: {delta_reg_final:.3f} MB -> {delta_comp_final:.3f} MB")
    print(f"PeakΔ  Regular vs Compressed: {delta_reg_peak:.3f} MB -> {delta_comp_peak:.3f} MB")
    if delta_reg_peak > 0:
        peak_savings = (delta_reg_peak - delta_comp_peak)
        print(f"Peak Savings: {peak_savings:.3f} MB ({peak_savings/delta_reg_peak*100:.1f}%)")

    # Tree internal estimation comparison
    if tree_stats_reg['node_count'] and tree_stats_comp['node_count']:
        reg_data = tree_stats_reg['state_arrays'] + tree_stats_reg['bit_arrays']
        comp_data = tree_stats_comp['state_arrays'] + tree_stats_comp['bit_arrays']
        if reg_data > 0:
            data_savings = reg_data - comp_data
            print(f"Data buffer savings (approx): {data_savings/1024/1024:.3f} MB ({data_savings/reg_data*100:.1f}%)")

    # Per-node unpacked data vs bit-packed proportion
    if tree_stats_comp['node_count']:
        bp_ratio = tree_stats_comp['bit_arrays'] / max(1.0, (tree_stats_comp['bit_arrays'] + tree_stats_comp['state_arrays']))
        print(f"Compressed variant bit-packed share of data: {bp_ratio*100:.1f}%")

    # Tracemalloc differential (Python objects only)
    total_py_reg = sum(stat.size for stat in snap_reg.statistics('filename')) / 1024 / 1024
    total_py_comp = sum(stat.size for stat in snap_comp.statistics('filename')) / 1024 / 1024
    print(f"Python-object alloc (tracemalloc) Regular vs Compressed: {total_py_reg:.3f} MB -> {total_py_comp:.3f} MB")
    if total_py_reg > 0:
        print(f"Python-layer savings: {(total_py_reg-total_py_comp):.3f} MB ({(total_py_reg-total_py_comp)/total_py_reg*100:.1f}%)")


if __name__ == "__main__":
    run_memory_comparison()
