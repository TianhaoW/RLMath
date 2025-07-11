import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

Point3D = namedtuple("Point3D", ["x", "y", "z"])

class GridSubset3DEnv(gym.Env):
    """Base environment for 3D grid-based subset selection tasks."""

    def __init__(self, l: int, m: int, n: int):
        """
        l: depth (z), m: height (y), n: width (x)
        """
        self.grid_shape = (l, m, n)
        self.action_space = spaces.Discrete(l * m * n)
        self.observation_space = spaces.Box(low=0, high=1, shape=(l, m, n), dtype=np.float32)

        self.state = np.zeros((l, m, n), dtype=np.float32)
        self.points = np.empty((0,3), dtype=int)  # k x 3
        self.terminated = False
        self.badpoint = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        l, m, n = self.grid_shape
        self.state.fill(0)
        self.points = np.empty((0,3), dtype=int)
        self.terminated = False
        self.badpoint = None
        return self.state.copy(), {}

    def decode_action(self, action: int) -> Point3D:
        """Maps flat action to (x,y,z)"""
        l, m, n = self.grid_shape
        z = action // (m * n)
        rem = action % (m * n)
        y = rem // n
        x = rem % n
        return Point3D(x, y, z)

    def encode_action(self, point: Point3D) -> int:
        """Maps (x,y,z) to flat action"""
        l, m, n = self.grid_shape
        return point.z * m * n + point.y * n + point.x

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Step called on terminated environment")
        point = self.decode_action(action)
        done, reward = self.add_point(point)
        self.terminated = done
        return self.state.copy(), reward, done, False, {}

    def self_play_add_point(self, point: Point3D, plot=True):
        if self.terminated:
            raise RuntimeError("Game already ended. Please reset first.")

        l, m, n = self.grid_shape
        if 0 <= point.z < l and 0 <= point.y < m and 0 <= point.x < n and not self.is_selected(point):
            done, _ = self.add_point(point)
            if done:
                print("game over")
                self.terminated = True
        else:
            print("Invalid point, already selected or out of bounds.")

        if plot:
            self.plot()

    def is_selected(self, point: Point3D) -> bool:
        return self.state[(point.z, point.y, point.x)] == 1

    def mark_selected(self, point: Point3D):
        self.state[(point.z, point.y, point.x)] = 1
        new_point = np.array([[point.x, point.y, point.z]])
        self.points = np.vstack([self.points, new_point])

    def add_point(self, point: Point3D):
        """Must be overridden by subclass"""
        raise NotImplementedError("Subclasses must implement add_point.")

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        if self.points.shape[0] > 0:
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='blue', s=60)
        if self.badpoint:
            ax.scatter([self.badpoint.x], [self.badpoint.y], [self.badpoint.z], c='red', s=100)

        l, m, n = self.grid_shape
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, m - 1)
        ax.set_zlim(0, l - 1)
        ax.set_aspect('equal')
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_zticks(range(l))

        # Draw 3 bold axis arrows
        arrow_length = max(l, m, n) * 0.8
        ax.quiver(0, 0, 0, arrow_length, 0, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, arrow_length, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, arrow_length, color='black', linewidth=2, arrow_length_ratio=0.1)

        ax.text(arrow_length, 0, 0, "X", fontsize=12, color='black')
        ax.text(0, arrow_length, 0, "Y", fontsize=12, color='black')
        ax.text(0, 0, arrow_length, "Z", fontsize=12, color='black')

        # Draw grid lines across the cube
        for x in range(n):
            for y in range(m):
                ax.plot([x, x], [y, y], [0, l - 1], color='lightgray', linewidth=0.5)
        for x in range(n):
            for z in range(l):
                ax.plot([x, x], [0, m - 1], [z, z], color='lightgray', linewidth=0.5)
        for y in range(m):
            for z in range(l):
                ax.plot([0, n - 1], [y, y], [z, z], color='lightgray', linewidth=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()

    @property
    def points_list(self):
        """Returns the points as a list of Point3D namedtuples"""
        return [Point3D(x.item(), y.item(), z.item()) for x, y, z in self.points]

    @property
    def points_count(self):
        """Direct number of points"""
        return self.points.shape[0]

class GridSubset3DEnvWithPriority(GridSubset3DEnv):
    """
    3D base environment that augments the observation with a second channel:
    the priority score of each grid point.

    Takes a priority_fn(Point3D, grid_shape) -> float
    """

    def __init__(self, l: int, m: int, n: int, priority_fn=None):
        self.priority_fn = priority_fn or self.default_priority
        self.priority_map = np.zeros((l, m, n), dtype=np.float32)
        self.obs = np.empty((2, l, m, n), dtype=np.float32)
        super().__init__(l, m, n)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(2, l, m, n), dtype=np.float32
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

    def _get_obs(self):
        self.obs[0][:] = self.state
        np.copyto(self.obs[1], self.priority_map, where=(self.priority_map != -np.inf))
        self.obs[1][self.priority_map == -np.inf] = -1.0
        return self.obs

    def _init_priority_map(self):
        l, m, n = self.grid_shape
        for z in range(l):
            for y in range(m):
                for x in range(n):
                    self.priority_map[z, y, x] = self.priority_fn(Point3D(x, y, z), self.grid_shape)

        # normalize to [0,1]
        max_p = np.max(self.priority_map)
        min_p = np.min(self.priority_map)
        if max_p > min_p:
            self.priority_map = (self.priority_map - min_p) / (max_p - min_p)

    def mark_selected(self, point: Point3D):
        self.state[(point.z, point.y, point.x)] = 1
        self.priority_map[(point.z, point.y, point.x)] = -np.inf
        new_point = np.array([[point.x, point.y, point.z]])
        self.points = np.vstack([self.points, new_point])

    def greedy_action_step(self, random=True, plot=True):
        valid_mask = (self.priority_map != -np.inf)
        if not np.any(valid_mask):
            if plot:
                self.plot()
            return -1

        if random:
            masked_map = np.where(valid_mask, self.priority_map, -np.inf)
            max_value = np.max(masked_map)
            zs, ys, xs = np.where(masked_map == max_value)
            idx = np.random.randint(len(xs))
            point = Point3D(int(xs[idx]), int(ys[idx]), int(zs[idx]))
        else:
            masked_flat = np.where(valid_mask, self.priority_map, -np.inf)
            flat_index = np.argmax(masked_flat)
            z, rem = divmod(flat_index, self.priority_map.shape[1] * self.priority_map.shape[2])
            y, x = divmod(rem, self.priority_map.shape[2])
            point = Point3D(int(x), int(y), int(z))

        self.self_play_add_point(point, plot=False)
        if plot:
            self.plot()
        return point

    def greedy_search(self, random=True):
        while True:
            result = self.greedy_action_step(random=random, plot=False)
            if result == -1:
                return len(self.points)

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if self.points.shape[0] > 0:
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='blue', s=60)


        zs, ys, xs = np.where(self.priority_map == -np.inf)
        for x, y, z in zip(xs, ys, zs):
            if not self.is_selected(Point3D(x, y, z)):
                ax.plot(x, y, z, 'o', color='red', markersize=6)

        l, m, n = self.grid_shape
        # Draw grid lines across the cube
        for x in range(n):
            for y in range(m):
                ax.plot([x, x], [y, y], [0, l - 1], color='lightgray', linewidth=0.5)
        for x in range(n):
            for z in range(l):
                ax.plot([x, x], [0, m - 1], [z, z], color='lightgray', linewidth=0.5)
        for y in range(m):
            for z in range(l):
                ax.plot([0, n - 1], [y, y], [z, z], color='lightgray', linewidth=0.5)


        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, m - 1)
        ax.set_zlim(0, l - 1)
        ax.set_aspect('equal')
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_zticks(range(l))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


    @staticmethod
    def default_priority(point: Point3D, grid_shape) -> float:
        return 0

