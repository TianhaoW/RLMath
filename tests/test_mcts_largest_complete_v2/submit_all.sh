#!/bin/bash
for start in {49..69}; do
    sbatch test_mcts_largest_complete_set.sub "$start"
done