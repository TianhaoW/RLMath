from src.envs.base_env import GridSubsetEnv, Point
from collections import defaultdict

def distance(p1: Point, p2: Point):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def area(p1: Point, p2: Point, p3: Point) -> float:
    return abs(p1.x * (p2.y - p3.y) +
               p2.x * (p3.y - p1.y) +
               p3.x * (p1.y - p2.y)) / 2.0

def distance(p1: Point, p2: Point):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

class NoIsoscelesEnv(GridSubsetEnv):
    def __init__(self, m, n):
        super().__init__(m, n)
        self.distance_map = {}

    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed, options)
        self.distance_map = {}
        return obs, {}

    def add_point(self, point: Point):
        if self.is_selected(point):
            return True, -10  # Already selected

        self.distance_map[point] = defaultdict(list)
        for p in self.points:
            d = distance(point, p)
            self.distance_map[p][d].append(point)
            self.distance_map[point][d].append(p)

            if len(self.distance_map[point][d]) >= 2 or len(self.distance_map[p][d]) >= 2:
                self.badpoint = point
                return True, 0  # Triangle formed

        self.mark_selected(Point(int(point.x), int(point.y)))
        self.points.append(Point(int(point.x), int(point.y)))
        return False, 1

# The difference here is that strict isosceles triangle should have area greater than 0
class NoStrictIsoscelesEnv(NoIsoscelesEnv):
    def add_point(self, point: Point):
        if self.is_selected(point):
            return True, -10  # Already selected

        self.distance_map[point] = defaultdict(list)
        for p in self.points:
            d = distance(point, p)
            self.distance_map[p][d].append(point)
            self.distance_map[point][d].append(p)

            if len(self.distance_map[p][d]) >= 2:
                p1 = self.distance_map[p][d][0]
                if area(point, p, p1) > 0:
                    self.badpoint = point
                    return True, 0

            if len(self.distance_map[point][d]) >= 2:
                p1 = self.distance_map[point][d][0]
                if area(p, point, p1) > 0:
                    self.badpoint = point
                    return True, 0

        self.mark_selected(Point(int(point.x), int(point.y)))
        self.points.append(Point(int(point.x), int(point.y)))
        return False, 1

