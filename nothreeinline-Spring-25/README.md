# nothreeinline-Spring-25

# Approach of Monte Carlo Tree Search (MCTS) and AlphaZero

Since the No Three in Line problem closely resembles a type of board game, we were reminded of AlphaZero, which has demonstrated excellent performance on Go-related problems. In this branch, I attempted to explore the bounds of the No Three in Line problem using MCTS and AlphaZero (MCTS-guided reinforcement learning).

## Background

Here's the best video of the explanation of MCTS algorithm I've found:

[Monte Carlo Tree Search](https://www.youtube.com/watch?v=UXW2yZndl7U)

This is a good AlphaZero algorithm hand-by-hand and step-by-step tutorial with source code:

[AlphaZero from Scratch – Machine Learning Tutorial](https://www.youtube.com/watch?v=wuSQpLinRB4&t=11603s)

[AlphaZeroFromScratch - Code](https://github.com/foersterrobert/AlphaZeroFromScratch)

Also, the [AlphaZero paper](https://arxiv.org/pdf/1712.01815) would be helpful.

## Current Progress

Initially, my work is simply the modification and adaptation of AlphaZero to the no-three-in-line problem. 

### "Game Class"

First thing I did is to write a "Game Class" which specifies the environment, valid actions, terminal condition, and value function of the no three in line. Thank to Tianhao's framework, we will have a standard, gym-compatible Game Class. Here, I just want to talk about two approaches of specifying valid actions, aka valid moves, and terminal condition.

#### _1. Any empty spot_

Valid action: set of empty spot.

Terminal condition: exist collinear triples.

(See [no3inlineClass](01_1_no3inlineClass.ipynb) and [Version 2.1](02_1_N3IL_MCTS.ipynb))

#### _2. No collinear triple spot_

Valid action: set of empty spot that is not collinear with any tuples on the board. 

Terminal condition: set of valid actions is empty.

This method significantly decreased the computational cost by avoiding redundant computation and reducing the action space. It roughly reduces the running time to 1/2~1/3, based on my observation (See [Version 2.2](02_2_N3IL_MCTS.ipynb)).

### Pure MCTS

#### [Version 2.2](02_2_N3IL_MCTS.ipynb)

This version found a valid configuration with 18 points in a 10 by 10 grid (the best it has found is actually 19, but I didn't save it, since the simulation involves some randomness) and 33 points in a 20 by 20 grid.

### MCTS + ResNet

#### [Version 7.1](07_1_N3IL_AlphaTweaks.ipynb)

This version is less effective than pure MCTS. For example, it only finds a valid configuration of 14 points in a 10 by 10 grid. 

### Parallel MCTS by Virtual Loss

Node value - 1, as a thread simulating on a node, to avoid multiple threads visit same node. ([Version 2.3.4](02_03_04_Pure_MCTS_Numba_Parallel_valid_move_simulate.ipynb))

### "Speed Up" with C++

With jit-compiled functions, the algorithm is faster 100 times than the pure python. ([Version 2.3](02_03_Pure_MCTS_Numba_test.ipynb))

### MCTS Along Priority - "TopN" method

Two types of priority:

1. [Number of collinear triples the point could be in.](08_04_MCTS_along_prority_topN_Get_MCTS.ipynb)
2. L-shaped priority.

## Results

20 valid points in 10x10 grid: [Version 2.3.2 Pure MCTS Parallel](02_03_02_Pure_MCTS_Numba_Parallel.ipynb)

MCTS along priority: [Version 8.4 MCTS along priority "TopN"](08_04_MCTS_along_prority_topN_Get_MCTS.ipynb)

## Next Steps/Questions

### L-shaped Priority

### Tree Reuse?

### MCGS?

Same as saving valid move at each configuration?

### AlphaZero with Transformer v.s. ResNet

ResNet (a type of CNN) is good at extracting the local features, while Transformer extracts global features.

We can consider Transformer or CNN-Transformer.