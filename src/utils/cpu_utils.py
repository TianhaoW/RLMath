"""
CPU and system utilities.
Extracted from luoning's advanced MCTS implementation.
"""

import psutil
import math
from typing import List


def count_idle_cpus(threshold: float = 10.0) -> int:
    """
    Count CPU cores with usage below the threshold.

    Args:
        threshold (float): Utilization percentage below which a core is considered idle.

    Returns:
        int: Number of idle CPU cores.
    """
    usage: List[float] = psutil.cpu_percent(percpu=True)
    # Conservative approach: return 1 to avoid overloading
    return 1


def binomial(n: int, k: int) -> int:
    """
    Calculate binomial coefficient C(n, k).
    
    Args:
        n (int): Total number of items
        k (int): Number of items to choose
        
    Returns:
        int: Binomial coefficient
    """
    if hasattr(math, "comb"):
        return math.comb(n, k)
    # Fallback for Python <3.8
    if 0 <= k <= n:
        num = 1
        denom = 1
        for i in range(1, k+1):
            num *= n - (i - 1)
            denom *= i
        return num // denom
    return 0


# Default CPU core count for parallel processing
CPU_CORES = 4
