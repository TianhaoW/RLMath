import sys
import os

import matplotlib.pyplot as plt
from src.envs import NoThreeCollinearEnvWithPriority, Point
import multiprocessing
import time

logical_cores = multiprocessing.cpu_count()
print(f"Logical cores: {logical_cores}")
N_JOBS = logical_cores - 1  # Reserve one core for the main process


def example_construction(n):
    return []

def sample_priority_fn(point, grid_size):
    return max(point.x, point.y)

def worker_create_config(n, construction, progress_dict):
    points = construction(n)
    progress_dict[n] = len(points)
    print(f"Config creation for n={n} completed with {len(points)} points.")
    return (n, points)

def create_config_for_ns(ns, construction):
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    for n in ns:
        progress_dict[n] = None

    args = [(n, construction, progress_dict) for n in ns]

    with multiprocessing.Pool(processes=min(N_JOBS, len(ns))) as pool:
        results = pool.starmap(worker_create_config, args)

    results.sort(key=lambda x: x[0])
    return {n: points for n, points in results}, progress_dict

def worker_run_greedy(n, starting_config, priority_fn, progress_dict):
    env = NoThreeCollinearEnvWithPriority(n, n, priority_fn)
    points = starting_config if starting_config is not None else []
    for x, y in points:
        env.self_play_add_point(Point(x, y), plot=False)
    env.greedy_search()
    progress_dict[n] = len(env.points)
    print(f"Greedy search for n={n} completed with {len(env.points)} points.")
    return (n, len(env.points), env.points)

def run_greedy_for_ns(ns, starting_config=None, priority_fn=None):
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    for n in ns:
        progress_dict[n] = None

    args = [(n, starting_config, priority_fn, progress_dict) for n in ns]

    with multiprocessing.Pool(processes=min(N_JOBS, len(ns))) as pool:
        results = pool.starmap(worker_run_greedy, args)

    results.sort(key=lambda x: x[0])
    all_counts = {n: count for n, count, _ in results}
    return results, all_counts, progress_dict

if __name__ == "__main__":
    nums = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241)

    configs, config_progress = create_config_for_ns(nums, example_construction)

    while any(v is None for v in config_progress.values()):
        print("Config creation progress:", dict(config_progress))
        time.sleep(0.5)
    print("Config creation completed:", dict(config_progress))

    results, greedy_sizes, greedy_progress = run_greedy_for_ns(nums, starting_config=None, priority_fn=None)

    while any(v is None for v in greedy_progress.values()):
        print("Greedy search progress:", dict(greedy_progress))
        time.sleep(0.5)
    print("Greedy search completed:", dict(greedy_progress))

    plt.figure(figsize=(16, 10))
    plt.title("Random Greedy Improvement of the conic p construction")
    plt.plot(greedy_sizes.keys(), greedy_sizes.values(), marker='o')
    plt.show()
