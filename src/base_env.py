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

    def self_play_add_point(self, point: Point):
        if self.terminated:
            raise RuntimeError("Game already ended. Please start a new game by calling reset() first.")

        m, n = self.grid_shape
        if 0 <= point.x < n and 0 <= point.y < m and not self.is_selected(point):
            done, _ = self.add_point(point)
            if done:
                print("game over")
                self.terminated = True
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

        ax.set_xlim(-0.5, n)
        ax.set_ylim(-0.5, m)
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_aspect('equal')
        plt.grid(False)
        plt.show()

    def is_selected(self, point: Point) -> bool:
        m, _ = self.grid_shape
        return self.state[(point.y, point.x)] == 1

    def mark_selected(self, point: Point):
        m, _ = self.grid_shape
        self.state[(point.y, point.x)] = 1