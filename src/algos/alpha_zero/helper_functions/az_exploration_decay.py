from numba import njit
import numpy as np

@njit(cache=True, nogil=True)
def exploration_decay_nb(x):  # Monotone-down from (0,1) to (1,0)
    # Cosine decay
    # return (np.cos(np.pi * x)+1)/2  # 100% exploration at start, 0% at end
    
    # Linear
    # return 1 - 0.7 * x   # Found optimal 4-point solution: 86/100 times (86.0%)
    # return 1 - x # 85/100 times (85.0%)

    # Square root (gentle early decay)
    # return 1 - 0.9 * np.sqrt(x) # 91/100 times (91.0%)
    # return 1 - 1 * np.sqrt(x) # 83/100 times (83.0%)
    # return 1 - 0.5 * np.sqrt(x) # 88/100 times (88.0%)
    # return 1 - 0.7 * np.sqrt(x) # 86%
    # return 1 - 0.8 * np.sqrt(x) # 92/100 times (92.0%)
    return 1 - 0.85 * np.sqrt(x)

    # Quadratic (faster decay)
    # return 1 - (x ** 2)

    # Exponential (custom normalization)
    # return ((np.exp(1)/(np.exp(1)-1))**2) * ((np.exp(-x)-np.exp(-1)) ** 2) # 85/100 times (85.0%)

    # Exponential fast (k=3)
    #k = 3.0
    # return (np.exp(-k * x) - np.exp(-k)) / (1 - np.exp(-k)) # 90/100 times (90.0%)

    # Exponential slow (k=1)
    # k = 1.0
    # return (np.exp(-k * x) - np.exp(-k)) / (1 - np.exp(-k)) # 86/100 times (86.0%)

    # Cosine decay
    # return 0.5 * (1 + np.cos(np.pi * x)) # 85/100 times (85.0%)

    # Rational decay
    # a = 1.0
    # return (1 - x) / (1 + a * x) # solution: 90/100 times (90.0%)

    # Logistic decay
    # k = 10.0
    # g0 = 1 / (1 + np.exp(k * (0 - 0.5)))
    # g1 = 1 / (1 + np.exp(k * (1 - 0.5)))
    # gx = 1 / (1 + np.exp(k * (x - 0.5)))
    # return (gx - g1) / (g0 - g1) # 86/100 times (86.0%)

    # Cubic decay
    # return 1 - x ** 3 # 91/100 times (91.0%)
    # return 1 - (0.9 * (x ** 3)) # 91/100 times (91.0%)