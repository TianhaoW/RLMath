import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

Point = namedtuple("Point", ["x", "y"])

class GridSubsetRemoveEnv(gym.Env):
    """
    Base environment for grid-based subset removal tasks.
    Starts with all points selected (state=1). Removes points to achieve constraints.
    """
    def __init__(self, m: int, n: int):
        self.grid_shape = (m, n)
        self.observation_space = spaces.Box(low=0, high=1, shape=(m, n), dtype=np.float32)
        self.action_space = spaces.Discrete(m * n)
        self.terminated = False
        self.state = None
        self.points = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, n = self.grid_shape
        self.state = np.ones((m, n), dtype=np.float32)
        self.points = [Point(x, y) for x in range(n) for y in range(m)]
        self.terminated = False
        return self.state.copy(), {}

    def decode_action(self, action: int) -> Point:
        m, n = self.grid_shape
        x, y = divmod(action, m)
        return Point(x, y)

    def encode_action(self, point: Point) -> int:
        m, n = self.grid_shape
        return point.x * m + point.y

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Step called on terminated environment")

        point = self.decode_action(action)
        done, reward = self.remove_point(point)

        obs = self.state.copy()
        self.terminated = done
        return obs, reward, done, False, {}

    def remove_point(self, point: Point):
        raise NotImplementedError("Subclasses must override `remove_point`.")

    def is_selected(self, point: Point) -> bool:
        return self.state[(point.y, point.x)] == 1

    def mark_unselected(self, point: Point):
        self.state[(point.y, point.x)] = 0

    def plot(self):
        m, n = self.grid_shape
        fig, ax = plt.subplots(figsize=(n, m))

        # Draw grid
        for x in range(n + 1):
            ax.axvline(x, color='lightgray', linewidth=1)
        for y in range(m + 1):
            ax.axhline(y, color='lightgray', linewidth=1)

        # Plot points
        for p in self.points:
            if self.is_selected(p):
                ax.plot(p.x, p.y, 'o', color='blue', markersize=12)

        ax.set_xlim(-0.5, n-0.5)
        ax.set_ylim(-0.5, m-0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_aspect('equal')
        plt.grid(False)
        plt.show()