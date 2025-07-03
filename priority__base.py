import matplotlib.pyplot as plt
from src.envs import NoThreeCollinearEnv, Point, NoThreeCollinearEnvWithPriority
import concurrent.futures
from src.utils import get_logger, parse_config

config = parse_config()
logger = get_logger("HJSW", config)

def conic_p_construction(p):
    points = []
    for x in range(p):
        for y in range(p):
            if (y-x**2) % p == 0:
                points.append(Point(x, y))

    return points

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

primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
          97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
          191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
          283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
          401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499)

def process_prime(p):
    env = NoThreeCollinearEnvWithPriority(2*p, 2*p)
    for point in HJSW_2p_construction(p, k=1):
        env.self_play_add_point(Point(point[0], point[1]), plot=False)
    env.greedy_search()
    improvement = len(env.points) - 3*(p-1)
    logger.info(f"p={p}: greedy improvement: {improvement}")
    return (p, improvement)

def main():# example
    improve = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_prime, primes))
    for p, improvement in results:
        improve[p] = improvement

    plt.figure(figsize=(16,10))
    plt.title("Random Greedy Improvement of the HJSW construction")
    plt.plot(improve.keys(), improve.values(), marker='o')
    plt.show()

if __name__ == "__main__":
    main()