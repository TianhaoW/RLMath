# MCTS Tree Visualization Feature

This document explains how to use the new tree visualization feature added to the MCTS implementation.

## Overview

The tree visualization feature uses [pyvis](https://pyvis.readthedocs.io/) to create interactive HTML visualizations of the MCTS search tree. Each node in the tree displays:

- Visit count
- Value sum  
- Average value
- UCB (Upper Confidence Bound) score
- Game state as an image

## Usage

### Enabling Tree Visualization

To enable tree visualization, set `tree_visualization: True` in your args dictionary:

```python
args = {
    'environment': 'N3il',
    'algorithm': 'MCTS',
    'n': 3,
    'num_searches': 100,
    'tree_visualization': True,  # Enable tree visualization
    # ... other args
}
```

### Interactive Mode

When tree visualization is enabled, after each MCTS search, you will be prompted:

```
Output action prob? (y/n):
```

- Type `y` and press Enter to see the action probabilities printed to the console
- Type `n` and press Enter to skip printing and continue to the next search

### Output Files

The tree visualization creates several types of output files:

1. **Multi-snapshot HTML**: A single HTML file containing all tree snapshots with navigation controls
   - Format: `mcts_tree_multi_n{grid_size}_points{final_points}_{timestamp}.html`
   - Allows switching between different game steps using buttons or arrow keys

2. **Individual snapshot HTML files**: Separate HTML files for each game step
   - Format: `mcts_tree_step{step_number}_n{grid_size}_{timestamp}.html`
   - Contains the complete interactive pyvis visualization for that step

3. **State images**: Game state visualizations are embedded as base64 images in the tree nodes

### File Locations

Visualization files are saved to the directory specified by the `figure_dir` argument:

```python
args = {
    'figure_dir': '/path/to/save/visualizations',
    # ... other args
}
```

If `figure_dir` is not specified, files are saved to the current directory.

## Examples

### Basic Usage

```python
from src.algos.mcts import evaluate

args = {
    'environment': 'N3il',
    'algorithm': 'MCTS',
    'n': 3,
    'C': 1.41,
    'num_searches': 50,
    'tree_visualization': True,
    'figure_dir': './tree_visualizations'
}

num_points = evaluate(args)
```

### Automated Testing (No User Input)

For automated testing, you can mock the input function:

```python
import builtins

# Mock input to always return 'n'
original_input = builtins.input
builtins.input = lambda prompt: 'n'

try:
    num_points = evaluate(args)
finally:
    builtins.input = original_input
```

## Visualization Features

### Tree Layout
- Hierarchical top-to-bottom layout
- Parent nodes at the top, children below
- Automatic spacing and positioning

### Node Information
Each node displays:
- **Visit Count**: Number of times the node was visited during MCTS
- **Value Sum**: Sum of all simulation values from this node
- **Average Value**: Value Sum / Visit Count
- **UCB Score**: Upper Confidence Bound used for node selection

### Game State Images
- Each node shows the current game state as a small grid image
- Blue dots indicate placed points
- Grid lines show the game board structure

### Navigation (Multi-snapshot view)
- **Previous/Next buttons**: Navigate between snapshots
- **Dropdown menu**: Jump directly to any snapshot
- **Keyboard shortcuts**: Use ← and → arrow keys to navigate
- **Snapshot info**: Shows current snapshot number and name

## Performance Considerations

- Tree visualization adds some computational overhead
- Image generation requires matplotlib figure creation for each node
- Large trees (many nodes) may result in large HTML files
- Consider reducing `num_searches` for testing with visualization enabled

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'pyvis'**
   - Install pyvis: `pip install pyvis`

2. **Empty visualization files**
   - Ensure `tree_visualization: True` is set in args
   - Check that MCTS is actually running (num_searches > 0)

3. **Permission errors when saving files**
   - Ensure the `figure_dir` path exists and is writable
   - Use absolute paths when possible

### Dependencies

The tree visualization feature requires:
- `pyvis` >= 0.3.0
- `matplotlib` (for image generation)
- `numpy` (already required by MCTS)

## Implementation Details

### Key Methods

- `MCTS._state_to_image_base64()`: Converts game state to base64 image
- `MCTS._get_node_label()`: Generates node labels with statistics
- `MCTS.tree_visualization()`: Creates pyvis network from MCTS tree
- `MCTS.save_multi_snapshot_html()`: Saves multi-snapshot HTML file

### Data Storage

Tree snapshots are stored in `MCTS.snapshots` as dictionaries containing:
- `name`: Human-readable snapshot name
- `network`: pyvis Network object
- `html`: Generated HTML content

This allows for flexible post-processing and custom visualization formats.
