from src.envs.base_env import GridSubsetEnv, Point, GridSubsetEnvWithPriority
from collections import defaultdict
from math import gcd
import numpy as np

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
        super().__init__(m, n)
        self.slope_map = defaultdict(set)  # maps Point p to set of slopes seen from p

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

        # Now add the new point and initialize its slope set
        for p in self.points:
            slope = reduced_slope(new_point, p)
            self.slope_map[new_point].add(slope)

        self.mark_selected(new_point)
        self.points.append(new_point)
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
        for x in range(n):
            for y in range(m):
                test = Point(x, y)
                if self.is_selected(test) or self.priority_map[y, x] == -np.inf:
                    continue
                # Invalidate if collinear with new_point and any selected point
                for cap in self.points:
                    if cap == new_point:
                        continue
                    if self._are_collinear(new_point, cap, test):
                        self.priority_map[y, x] = -np.inf
                        break

    def _are_collinear(self, p1: Point, p2: Point, p3: Point) -> bool:
        # Avoid exact slope check by using determinant
        return (p1.x * (p2.y - p3.y) +
                p2.x * (p3.y - p1.y) +
                p3.x * (p1.y - p2.y)) == 0