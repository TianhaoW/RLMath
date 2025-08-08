#!/bin/bash
for start in {3..22}; do
    sbatch test_mcts_smallest_complete_set.sub "$start" 20
done