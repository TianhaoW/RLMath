from src.envs.base_env import GridSubsetEnv, Point, GridSubsetEnvWithPriority
from src.envs.base_env_remove import GridSubsetRemoveEnv
from collections import defaultdict
from itertools import combinations
from math import gcd
import copy
import numpy as np
import random
import matplotlib.pyplot as plt

def reduced_slope(p1: Point, p2: Point):
    """
    Return the slope from p1 to p2 as a reduced fraction tuple: (dy, dx)
    Vertical → (1, 0), horizontal → (0, 1), same point → (0, 0)
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    if dx == 0 and dy == 0:
        return (0, 0)
    g = gcd(dy, dx) if dx != 0 or dy != 0 else 1
    return (dy // g, dx // g)

class NoThreeCollinearEnv(GridSubsetEnv):
    """
    Grid environment where no three selected points can lie on the same line.
    Efficient implementation using slope hashing.
    """
    def __init__(self, m, n):
        self.slope_map = defaultdict(set)  # maps Point p to set of slopes seen from p
        super().__init__(m, n)


    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed, options)
        self.slope_map = defaultdict(set)
        return obs, {}

    def add_point(self, point: Point):
        if self.is_selected(point):
            return True, -10

        new_point = Point(int(point.x), int(point.y))

        for p in self.points:
            slope = reduced_slope(p, new_point)
            if slope in self.slope_map[p]:
                self.badpoint = new_point
                return True, 0
            self.slope_map[p].add(slope)
            # basically, (p1,p2) has the positive slope, and (p2,p1) has the negative slope. When adding new points,
            # we check both against the slope dictionary of p1, and slope dictionary of p2
            self.slope_map[new_point].add((-slope[0], -slope[1]))

        # This was the old implementation. To be deleted soon.
        # for p in self.points:
        #     slope = reduced_slope(new_point, p)
        #     self.slope_map[new_point].add(slope)

        self.mark_selected(new_point)
        self.points.append(new_point)
        return False, 1

# This is not fast
class FastNoThreeCollinearEnv(GridSubsetEnv):
    """
    Uses batch determinant with saved lines(wedge product) computation to accelerate. However, this is not faster.

    The issue is that there are k^2/2 number of lines, which makes checking new point slow compared to the saved slope method.
    For each line, we need to compute a new dot product. However, for the slope method, for each point,
    we only need to compute the slope, and check if it is in the dictionary
    """
    def __init__(self, m: int, n: int):
        super().__init__(m, n)
        self.points_array = np.empty((0,2), dtype=np.int32)  # stores (x,y)
        self.lines = np.empty((0,3), dtype=np.int32)         # stores (a,b,c)

    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed, options)
        self.points_array = np.empty((0,2), dtype=np.int32)
        self.lines = np.empty((0,3), dtype=np.int32)
        return obs, {}

    def add_point(self, point: Point):
        if self.is_selected(point):
            return True, -10

        x, y = int(point.x), int(point.y)

        # Check if point lies on any existing line
        if self.lines.shape[0] > 0:
            dot_products = self.lines @ np.array([x, y, 1], dtype=np.int32)
            if np.any(dot_products == 0):
                self.badpoint = point
                return True, 0

        # Mark point in state grid
        self.mark_selected(point)
        self.points.append(point)

        # Update points array
        new_point = np.array([[x, y]], dtype=np.int32)
        if self.points_array.size == 0:
            self.points_array = new_point
        else:
            x1 = self.points_array[:, 0]
            y1 = self.points_array[:, 1]
            x2 = x
            y2 = y

            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            new_lines = np.stack([a, b, c], axis=1)

            self.lines = np.vstack([self.lines, new_lines])
            self.points_array = np.vstack([self.points_array, new_point])

        return False, 1


class NoThreeCollinearEnvWithPriority(GridSubsetEnvWithPriority):
    def __init__(self, m: int, n: int, priority_fn=None):
        super().__init__(m, n, priority_fn)

    def add_point(self, point: Point):
        # If we cannot add this point, then the game ends
        if self.priority_map[point.y, point.x] == -np.inf:
            self.badpoint = point
            return True, 0

        # Else, we add the point, and update the state and priority
        new_point = Point(int(point.x), int(point.y))
        self.mark_selected(new_point)
        self.points.append(new_point)
        self._invalidate_collinear_points(new_point)

        return False, 1

    def _invalidate_collinear_points(self, new_point: Point):
        m, n = self.grid_shape
        num_existing = len(self.points)
        if num_existing == 0:
            return

        # Turn existing points into numpy array (k,2) k is the number of existing points
        existing_points = np.array([[p.x, p.y] for p in self.points if p != new_point], dtype=np.int32)

        # Build full test grid (mn,2)
        X, Y = np.meshgrid(np.arange(n), np.arange(m))
        test_points = np.stack([X.ravel(), Y.ravel()], axis=1)

        # Remove already selected or -inf from priority_map
        mask_valid = (self.state[Y.ravel(), X.ravel()] == 0) & (self.priority_map[Y.ravel(), X.ravel()] != -np.inf)
        test_points = test_points[mask_valid]

        # Now for each existing point, compute determinant batch
        for cap in existing_points:
            # determinants: shape (N,)
            dets = (new_point.x * (cap[1] - test_points[:, 1]) +
                    cap[0] * (test_points[:, 1] - new_point.y) +
                    test_points[:, 0] * (new_point.y - cap[1]))
            # Invalidate where determinant == 0 (collinear)
            collinear_mask = (dets == 0)
            to_invalidate = test_points[collinear_mask]
            self.priority_map[to_invalidate[:, 1], to_invalidate[:, 0]] = -np.inf

    # This function is no longer used after vectorized version of the priority_map. To be deleted
    def _are_collinear(self, p1: Point, p2: Point, p3: Point) -> bool:
        return (p1.x * (p2.y - p3.y) +
                p2.x * (p3.y - p1.y) +
                p3.x * (p1.y - p2.y)) == 0

class NoThreeInLineRemovalEnv(GridSubsetRemoveEnv):
    """
    Environment that starts full. Agent removes points to achieve no three on a line.
    """
    def __init__(self, m, n):
        self.triples_set = None
        self.triple_counter = None
        self.triple_index = None
        super().__init__(m, n)

    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed, options)
        self.triples_set, self.triple_counter, self.triple_index = self.build_triples_vectorized()
        # if sum(self.triple_counter) == 0:
        #     self.terminated = True
        return obs, {}

    def reset_with_points(self, points):
        m, n =self.grid_shape
        self.state = np.zeros((m, n), dtype=np.float32)
        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])
        self.state[ys, xs] = 1

        self.points = points
        self.terminated = False
        self.triples_set, self.triple_counter, self.triple_index = self.build_triples_vectorized(points)
        if sum(self.triple_counter) == 0:
            self.terminated = True

    # TODO, support this form of resets with points. The code is not correct yet. Write a separate set_env function.
    # May need to change self_state, self_
    def build_triples_vectorized(self, points=None):
        """
        points: list of Point
        returns:
          triples_set: set[frozenset]
          triple_counter: dict[Point, int]
          triple_index: dict[Point, set[frozenset]]
        """
        if not points:
            points = self.points
        triples_set = set()
        triple_counter = defaultdict(int)
        triple_index = defaultdict(set)

        all_points = np.array([[p.x, p.y] for p in points], dtype=np.int32)
        num_points = len(points)

        for i, (p1, p2) in enumerate(combinations(points, 2)):

            # Vectorized over remaining points
            mask = np.ones(num_points, dtype=bool)
            idx1 = points.index(p1)
            idx2 = points.index(p2)
            mask[idx1] = False
            mask[idx2] = False
            remaining_points = all_points[mask]

            # determinants: shape (N_remaining,)
            dets = (p1.x * (p2.y - remaining_points[:, 1])
                    + p2.x * (remaining_points[:, 1] - p1.y)
                    + remaining_points[:, 0] * (p1.y - p2.y))

            # collect triples
            collinear_idx = np.where(dets == 0)[0]
            for idx in collinear_idx:
                p3_idx = np.arange(num_points)[mask][idx]
                p3 = points[p3_idx]
                t = frozenset([p1, p2, p3])
                if t not in triples_set:
                    triples_set.add(t)
                    for p in t:
                        triple_counter[p] += 1
                        triple_index[p].add(t)

        return triples_set, triple_counter, triple_index

    def remove_point(self, point: Point):
        # if the point is already removed, we stop immediately
        if not self.is_selected(point):
            return False, -10

        # mark removed
        self.points.remove(point)
        self.mark_unselected(point)

        # TODO, check the implementation.
        # update triples
        for t in self.triple_index[point]:
            if t in self.triples_set:
                self.triples_set.remove(t)
                for p in t:
                    self.triple_counter[p] -= 1
        # remove index entry
        del self.triple_index[point]

        if sum(self.triple_counter.values()) == 0:
            return True, 0
        return False, 1

    def greedy_remove_step(self, rand=True, plot=True):
        if self.terminated:
            return True
        if rand:
            max_count = max(self.triple_counter.values())
            max_points = [p for p, count in self.triple_counter.items() if count == max_count]
            max_point = random.choice(max_points)
        else:
            max_point, max_count = max(self.triple_counter.items(), key=lambda item: item[1])
        terminated, _ = self.remove_point(max_point)
        print("removing", max_point, "with count", max_count)
        if plot:
            self.plot_heatmap()

        return terminated


    def greedy_remove(self, random=True):
        terminated = False
        while not terminated:
            terminated = self.greedy_remove_step(rand=random, plot=False)

    def plot_heatmap(self):
        m, n = self.grid_shape
        fig, ax = plt.subplots(figsize=(n, m))

        # Build triple heatmap
        triple_map = np.zeros((m, n), dtype=np.float32)
        for p, count in self.triple_counter.items():
            triple_map[p.y, p.x] = count

        # Avoid color scale collapsing on all-zero
        finite_vals = triple_map[triple_map != 0]
        min_val = np.min(finite_vals) if finite_vals.size > 0 else 0
        max_val = np.max(finite_vals) if finite_vals.size > 0 else 1
        if np.all(triple_map == 0):
            triple_map[0,0] = 0  # ensure at least one spot for consistent colorbar

        # Plot heatmap
        im = ax.imshow(triple_map, origin='lower', cmap='hot', alpha=0.4, vmin=min_val, vmax=max_val)

        # Draw grid lines
        for x in range(n + 1):
            ax.axvline(x - 0.5, color='lightgray', linewidth=1)
        for y in range(m + 1):
            ax.axhline(y - 0.5, color='lightgray', linewidth=1)

        # Plot selected points
        for p in self.points:
            if not self.is_selected(p):
                ax.plot(p.x, p.y, 'o', color='red', markersize=12)

        # Text counts
        for y in range(m):
            for x in range(n):
                if triple_map[y,x] > 0:
                    ax.text(x, y, f"{triple_map[y,x]:.0f}", ha='center', va='center', fontsize=9, color='black')

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, m - 0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_aspect('equal')
        plt.grid(False)
        plt.title("Triple Counter Heatmap")
        plt.colorbar(im, ax=ax, label='# Collinear Triples')
        plt.show()

class NoThreeInLineDominatingEnv(NoThreeCollinearEnvWithPriority):
    '''
    The difference of this environment with the NoThreeCollinearEnvWithPriority is that it allows add points
    even though it creates collinear triple. It will terminate if all points are covered.
    '''

    def __init__(self, m: int, n: int, priority_fn=None):
        super().__init__(m, n, priority_fn)

    def add_point(self, point: Point):
        # we allow add points even though it creates colinear triple
        # if self.priority_map[point.y, point.x] == -np.inf:
        #     self.badpoint = point
        #     return True, 0

        if self.is_selected(point):
            return True, -10

        # Else, we add the point, and update the state and priority
        new_point = Point(int(point.x), int(point.y))
        self.mark_selected(new_point)
        self.points.append(new_point)
        self._invalidate_collinear_points(new_point)

        if np.all(self.priority_map==-np.inf):
            return True, 0

        return False, 1

    def plot(self):
        m, n = self.grid_shape
        fig, ax = plt.subplots(figsize=(n, m))

        # Draw grid lines
        for x in range(n + 1):
            ax.axvline(x - 0.5, color='lightgray', linewidth=1)
        for y in range(m + 1):
            ax.axhline(y - 0.5, color='lightgray', linewidth=1)

        # Plot selected points
        for p in self.points:
            ax.plot(p.x, p.y, 'o', color='blue', markersize=12)

        ys, xs = np.where(self.priority_map != -np.inf)
        for x, y in zip(xs, ys):
            if Point(x,y) not in self.points:
                ax.plot(x, y, 'o', color='green', markersize=12)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, m - 0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_aspect('equal')
        plt.grid(False)
        plt.title("Grid with selected points and remaining points")

        plt.show()