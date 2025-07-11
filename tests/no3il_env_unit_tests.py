import time
import numpy as np
from src.envs import Point, FastNoThreeCollinearEnv, NoThreeCollinearEnvWithPriority, NoThreeCollinearEnv

def test_add_point_valid(Env):
    env = Env(10, 10)

    # Adding (0,0), (1,1), (2,0) should NOT be collinear
    done, reward = env.add_point(Point(0, 0))
    assert not done and reward == 1
    done, reward = env.add_point(Point(1, 1))
    assert not done and reward == 1
    done, reward = env.add_point(Point(2, 0))
    assert not done and reward == 1

    # Check internal state
    assert env.state[0,0] == 1
    assert env.state[1,1] == 1
    assert env.state[0,2] == 1
    assert len(env.points) == 3

def test_add_point_invalid(Env):
    env = Env(10, 10)

    # Adding (0,0), (1,1), (2,2) should be collinear
    done, reward = env.add_point(Point(0, 0))
    assert not done
    done, reward = env.add_point(Point(1, 1))
    assert not done
    done, reward = env.add_point(Point(2, 2))
    assert done
    assert reward == 0
    assert env.badpoint == Point(2, 2)

def test_multiple_pairs(Env):
    env = Env(10, 10)
    points = [Point(0,0), Point(1,2), Point(2,5), Point(3,1)]
    for p in points:
        done, reward = env.add_point(p)
        assert not done

    # Now adding (4,8) should fail, since (0,0),(2,4),(4,8) are collinear
    done, reward = env.add_point(Point(4,8))
    assert done
    assert reward == 0
    assert env.badpoint == Point(4,8)

def test_reset_and_reuse(Env):
    env = Env(10, 10)
    env.add_point(Point(0, 0))
    env.add_point(Point(1, 1))
    env.reset()
    assert len(env.points) == 0
    assert np.all(env.state == 0)

def HJSW_2p_construction(p, k=1):
    points = []
    # Use integer division for half intervals
    half_p = (p - 1) // 2
    half_p_plus = (p + 1) // 2

    # Loop over x in extended range, with shifted origin to avoid floats
    for x in range(-half_p, p + half_p + 1):
        for y in range(2 * p):
            if (x * y) % p != k:
                continue

            # Check which block it falls into
            if (
                # A blocks
                (0 * p < x <= 0 * p + half_p and 1 * p + half_p_plus <= y < 2 * p) or
                (1 * p < x <= 1 * p + half_p and 0 * p + half_p_plus <= y < 1 * p) or
                (1 * p < x <= 1 * p + half_p and 1 * p + half_p_plus <= y < 2 * p)
            ):
                points.append((x+half_p, y))
                continue

            if (
                # B blocks
                (0 * p + half_p_plus <= x < 1 * p and 1 * p + half_p_plus <= y < 2 * p) or
                (-1 * p + half_p_plus <= x < 0 * p and 0 * p + half_p_plus <= y < 1 * p) or
                (-1 * p + half_p_plus <= x < 0 * p and 1 * p + half_p_plus <= y < 2 * p)
            ):
                points.append((x+half_p, y))
                continue

            if (
                # C blocks
                (0 * p < x <= 0 * p + half_p and 0 * p < y <= 0 * p + half_p) or
                (1 * p < x <= 1 * p + half_p and 0 * p < y <= 0 * p + half_p) or
                (1 * p < x <= 1 * p + half_p and 1 * p < y <= 1 * p + half_p)
            ):
                points.append((x+half_p, y))
                continue

            if (
                # D blocks
                (0 * p + half_p_plus <= x < 1 * p and 0 * p < y <= 0 * p + half_p) or
                (-1 * p + half_p_plus <= x < 0 * p and 0 * p < y <= 0 * p + half_p) or
                (-1 * p + half_p_plus <= x < 0 * p and 1 * p < y <= 1 * p + half_p)
            ):
                points.append((x+ half_p, y))

    return points

def HJSW_construction_test(Env, p):
    env = Env(2*p, 2*p)
    points = HJSW_2p_construction(p)
    start = time.perf_counter()
    for point in points:
        done, reward = env.add_point(Point(point[0], point[1]))
        assert not done and reward == 1
    end = time.perf_counter()
    print(f"HJSW_construction_test on {2*p}x{2*p} took {end - start:.4f} seconds")

if __name__ == "__main__":
    # select environment you want to test here

    # Env = NewNoThreeCollinearEnvWithPriority
    # Env = NoThreeCollinearEnvWithPriority
    Env = NoThreeCollinearEnv
    # Env = FastNoThreeCollinearEnv
    test_add_point_valid(Env)
    test_add_point_invalid(Env)
    test_multiple_pairs(Env)
    test_reset_and_reuse(Env)
    HJSW_construction_test(Env, 499)
    print("All tests passed!")