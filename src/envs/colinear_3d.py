from collections import defaultdict
from math import gcd
from functools import reduce
from .base_env_3d import GridSubset3DEnv, GridSubset3DEnvWithPriority, Point3D
import numpy as np

def gcd3(a, b, c):
    return reduce(gcd, (a, b, c))

def reduced_direction(p1, p2):
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    if dx == dy == dz == 0:
        return (0,0,0)
    g = gcd3(dx, dy, dz) or 1
    return (dx // g, dy // g, dz // g)


class NoThreeCollinear3DEnv(GridSubset3DEnv):
    """
    Environment where no three selected points can lie on the same line in 3D.
    Uses slope hashing via reduced direction vectors for efficiency.
    """

    def __init__(self, l, m, n):
        self.slope_map = defaultdict(set)
        super().__init__(l, m, n)


    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.slope_map.clear()
        return obs, info

    def add_point(self, point: Point3D):
        p_new = np.array([point.x, point.y, point.z], dtype=int)

        if self.is_selected(point):
            return True, -10

        for p in self.points:
            p_tuple = tuple(p)
            direction = reduced_direction(p, p_new)
            if direction in self.slope_map[p_tuple]:
                self.badpoint = point
                return True, 0  # terminate with zero reward
            # update slope_map
            self.slope_map[p_tuple].add(direction)
            reverse_direction = tuple(-np.array(direction))
            self.slope_map[tuple(p_new)].add(reverse_direction)

        self.mark_selected(point)
        return False, 1

class NoThreeCollinear3DEnvWithPriority(GridSubset3DEnvWithPriority):
    """
    Environment where no three selected points can lie on the same line in 3D,
    with a priority function to guide greedy search.
    """

    def __init__(self, l: int, m: int, n: int, priority_fn=None):
        super().__init__(l, m, n, priority_fn)

    def add_point(self, point: Point3D):
        if self.priority_map[point.z, point.y, point.x] == -np.inf:
            self.badpoint = point
            return True, 0

        new_point = Point3D(int(point.x), int(point.y), int(point.z))
        self.mark_selected(new_point)
        self._invalidate_collinear_points(new_point)

        return False, 1

    def _invalidate_collinear_points(self, new_point: Point3D):
        l, m, n = self.grid_shape
        num_existing = len(self.points)
        if num_existing == 0:
            return

        existing_points = np.array([p for p in self.points if not np.array_equal(p, [new_point.x, new_point.y, new_point.z])],
                                   dtype=np.int32)
        if existing_points.shape[0] == 0:
            return

        # Build full test grid (N,3)
        Z, Y, X = np.meshgrid(np.arange(l), np.arange(m), np.arange(n), indexing='ij')
        test_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        mask_valid = (self.state[Z.ravel(), Y.ravel(), X.ravel()] == 0) & \
                     (self.priority_map[Z.ravel(), Y.ravel(), X.ravel()] != -np.inf)
        test_points = test_points[mask_valid]

        # Now for each existing point, compute cross product batch to check collinearity
        new_vec = np.array([new_point.x, new_point.y, new_point.z], dtype=np.int32)

        for cap in existing_points:
            vec1 = cap - new_vec  # shape (3,)
            vec2 = test_points - new_vec[None, :]  # shape (N,3)
            crosses = np.cross(vec1, vec2)  # shape (N,3)
            collinear_mask = np.all(crosses == 0, axis=1)
            to_invalidate = test_points[collinear_mask]
            self.priority_map[to_invalidate[:, 2], to_invalidate[:, 1], to_invalidate[:, 0]] = -np.inf

    @staticmethod
    def default_priority(point: Point3D, grid_shape) -> float:
        return 0