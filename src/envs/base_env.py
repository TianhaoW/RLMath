import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

Point = namedtuple("Point", ["x", "y"])

class GridSubsetEnv(gym.Env):
    """Base environment for grid-based subset selection tasks"""

    def __init__(self, m: int, n: int):
        self.grid_shape = (m, n)

        # Always output raw (m, n) grid state.
        # Model-specific encoding (flattening, channel dim, etc.) is handled by the model itself.
        self.observation_space = spaces.Box(low=0, high=1, shape=(m, n), dtype=np.float32)
        self.action_space = spaces.Discrete(m * n)
        self.terminated = False
        self.state = None
        self.badpoint = None
        self.points = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, n = self.grid_shape
        self.state = np.zeros((m, n), dtype=np.float32)

        self.points = []
        self.terminated = False
        self.badpoint = None
        return self.state.copy(), {}

    def decode_action(self, action: int) -> Point:
        m, n = self.grid_shape
        x, y = divmod(action, m)
        return Point(x, y)

    # Action encoding is column-major: (x, y) → x * m + y
    def encode_action(self, point: Point) -> int:
        m, n = self.grid_shape
        return point.x * m + point.y

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Step called on terminated environment")

        point = self.decode_action(action)
        done, reward = self.add_point(point)

        obs = self.state.copy()
        self.terminated = done
        return obs, reward, done, False, {}

    def self_play_add_point(self, point: Point, plot=True):
        if self.terminated:
            raise RuntimeError("Game already ended. Please start a new game by calling reset() first.")

        m, n = self.grid_shape
        if 0 <= point.x < n and 0 <= point.y < m and not self.is_selected(point):
            done, _ = self.add_point(point)
            if done:
                print("game over")
                self.terminated = True
            if plot:
                self.plot()
        else:
            print("adding invalid point, please try again")

    def add_point(self, point: Point):
        raise NotImplementedError("Subclasses must override `add_point`.")

    def plot(self):
        m, n = self.grid_shape
        fig, ax = plt.subplots(figsize=(n, m))

        # Draw grid lines
        for x in range(n + 1):
            ax.axvline(x, color='lightgray', linewidth=1)
        for y in range(m + 1):
            ax.axhline(y, color='lightgray', linewidth=1)

        # Plot points
        for p in self.points:
            ax.plot(p.x, p.y, 'o', color='blue', markersize=12)

        if self.badpoint:
            ax.plot(self.badpoint.x, self.badpoint.y, 'o', color='red', markersize=12)

        ax.set_xlim(-0.5, n-0.5)
        ax.set_ylim(-0.5, m-0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_aspect('equal')
        plt.grid(False)
        plt.show()

    def is_selected(self, point: Point) -> bool:
        return self.state[(point.y, point.x)] == 1

    def mark_selected(self, point: Point):
        self.state[(point.y, point.x)] = 1


class GridSubsetEnvWithPriority(GridSubsetEnv):
    """
    Base environment that augments the observation with a second channel:
    the priority score of each grid point.

    This class will take a priority_fn(Point, grid_size) as input.
    """

    def __init__(self, m: int, n: int, priority_fn=None):
        # initialize the priority map
        self.priority_fn = priority_fn or self.default_priority
        self.priority_map = np.zeros((m, n), dtype=np.float32)
        super().__init__(m, n)

        # the super().init will call the reset() function. We initialize the priority_map in the reset() function.

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, m, n), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self._init_priority_map()

        return self._get_obs(), info

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Step called on terminated environment")

        point = self.decode_action(action)
        done, reward = self.add_point(point)

        self.terminated = done
        return self._get_obs(), reward, done, False, {}

    # implement the greedy algorithm
    def greedy_action_step(self, random=True, plot=True):
        '''
        :param random: If True, this will return a random point with the highest priority. If false, it will return
        the first point with the highest priority.
        :param plot: If True, this will draw a plot
        :return: the picked point. If no points are available, return -1
        '''
        max_value = np.max(self.priority_map)
        if max_value == -np.inf:
            if plot:
                self.plot()
            return -1
        points = np.where(self.priority_map == max_value)
        if random:
            index = np.random.randint(len(points[0]))
        else:
            index = 0
        point = Point(int(points[1][index]), int(points[0][index]))
        self.self_play_add_point(point, plot)
        return point

    def greedy_search(self, random=True):
        while True:
            if(self.greedy_action_step(random=random, plot=False)==-1):
                return len(self.points)

    def _get_obs(self):
        """Returns stacked observation: [state, priority_map]"""
        return np.stack([self.state, np.nan_to_num(self.priority_map, neginf=-1.0)], axis=0)

    def _init_priority_map(self):
        m, n = self.grid_shape
        for x in range(n):
            for y in range(m):
                self.priority_map[y, x] = self.priority_fn(Point(x, y), self.grid_shape)

        # normalize the priority
        max_p = np.max(self.priority_map)
        min_p = np.min(self.priority_map)
        if max_p > min_p:
            self.priority_map = (self.priority_map - min_p) / (max_p - min_p)

    def mark_selected(self, point: Point):
        self.state[(point.y, point.x)] = 1
        self.priority_map[point.y, point.x] = -np.inf

    def plot(self):
        m, n = self.grid_shape
        fig, ax = plt.subplots(figsize=(n, m))

        # Plot the priority heatmap
        heatmap = np.copy(self.priority_map)
        # Replace -inf with min value for display
        finite_vals = heatmap[np.isfinite(heatmap)]
        min_val = np.min(finite_vals) if finite_vals.size > 0 else 0
        heatmap[~np.isfinite(heatmap)] = min_val

        ax.imshow(heatmap, origin='lower', cmap='viridis', alpha=0.4)

        # Draw grid lines
        for x in range(n + 1):
            ax.axvline(x - 0.5, color='lightgray', linewidth=1)
        for y in range(m + 1):
            ax.axhline(y - 0.5, color='lightgray', linewidth=1)

        # Plot selected points
        for p in self.points:
            ax.plot(p.x, p.y, 'o', color='blue', markersize=12)

        # Plot bad point (if any)
        # if self.badpoint:
        #     ax.plot(self.badpoint.x, self.badpoint.y, 'o', color='red', markersize=12)

        ys, xs = np.where(self.priority_map == -np.inf)
        for x, y in zip(xs, ys):
            if Point(x,y) not in self.points:
                ax.plot(x, y, 'o', color='red', markersize=12)

        ys, xs = np.where(np.isfinite(self.priority_map))
        for x,y in zip(xs, ys):
            ax.text(x, y, f"{self.priority_map[y, x]:.2f}")

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, m - 0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_aspect('equal')
        plt.grid(False)
        plt.title("Grid with Priority Heatmap")
        plt.colorbar(ax.imshow(heatmap, origin='lower', cmap='viridis', alpha=0.4),
                     ax=ax, label='Priority')
        plt.show()

    @staticmethod
    def default_priority(point: Point, grid_shape) -> float:
        """no priority"""
        return 0