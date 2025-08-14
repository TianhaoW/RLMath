#!/bin/bash
for start in {3..20}; do
    sbatch test_mcts_largest_complete_set.sub "$start" 150
done