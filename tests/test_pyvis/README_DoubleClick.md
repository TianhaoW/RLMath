# Double-Click Expand/Collapse & Compact Layout for MCTS Tree Visualization

## Overview
This enhanced implementation provides interactive double-click expand/collapse functionality and compact layout features for PyVis-generated MCTS tree visualizations. By default, only the root node and its immediate children (level 1) are visible, with automatic re-layout and tight packing when nodes are collapsed.

## Recent Updates
- **Compact Layout**: Removed fixed node positions for dynamic layout
- **Auto Re-layout**: Automatic compact re-layout after expand/collapse operations
- **Tight Packing**: Visible nodes pack closer together when others are hidden
- **Configurable Spacing**: Tunable level separation and node spacing parameters

## Features

### 1. Initial Visibility Control
- **Default Behavior**: Only root (level 0) and level 1 nodes are visible initially
- **Configurable**: Set `args['initial_visible_depth']` to control initial depth (default: 1)
- **Deep Trees**: Nodes beyond the initial depth are automatically hidden

### 2. Interactive Expand/Collapse with Auto Re-layout
- **Double-Click**: Double-click any visible node to toggle its children's visibility
- **Smart Toggling**: If children are hidden, they become visible; if visible, they become hidden
- **Edge Synchronization**: Connecting edges are automatically hidden/shown with their nodes
- **Group Behavior**: All children of a node are toggled together as a group
- **Auto Compact**: Layout automatically re-compacts after each toggle (~0.3s animation)
- **Resize Handling**: Layout adjusts on window resize

### 3. Compact Layout Configuration
- **Level Separation**: Set `args['level_separation']` (default: 90) for vertical spacing
- **Node Spacing**: Set `args['node_spacing']` (default: 60) for horizontal spacing  
- **Dynamic Positioning**: No fixed coordinates - vis.js computes optimal positions
- **Visible-Only Fitting**: Viewport fits to visible nodes only after operations

### 4. Recursive Mode (Optional)
- **Configuration**: Set `args['toggle_children_recursive'] = True` for recursive behavior
- **Behavior**: Instead of toggling only direct children, toggles entire descendant subtree
- **Use Case**: Useful for quickly expanding/collapsing large tree branches

## Configuration Parameters

Add these optional parameters to your MCTS args dictionary:

```python
args = {
    # ... existing parameters ...
    'initial_visible_depth': 1,        # Show levels 0-1 initially (default: 1)
    'toggle_children_recursive': False, # Toggle subtrees instead of direct children (default: False)
    'level_separation': 90,             # Vertical spacing between levels (default: 90)
    'node_spacing': 60,                 # Horizontal spacing between nodes (default: 60)
    'tree_visualization': True,         # Enable tree visualization (required)
}
```

## Usage Examples

### Basic Usage (Default Behavior)
```python
from src.algos.mcts import MCTS
from src.envs import N3il, supnorm_priority_array

args = {
    'environment': 'N3il',
    'n': 3,
    'tree_visualization': True,
    # ... other required parameters ...
}

# Initialize
priority_grid = supnorm_priority_array(args['n'])
game = N3il(grid_size=(args['n'], args['n']), args=args, priority_grid=priority_grid)
mcts = MCTS(game, args)

# Run search - creates tree visualization with double-click functionality
state = game.get_initial_state()
action_probs = mcts.search(state)
```

### Advanced Configuration
```python
args = {
    'environment': 'N3il',
    'n': 3,
    'tree_visualization': True,
    'initial_visible_depth': 2,        # Show levels 0-2 initially
    'toggle_children_recursive': True,  # Enable recursive expand/collapse
    # ... other parameters ...
}
```

### Comprehensive Visualization
```python
# After running multiple trials, generate comprehensive HTML
web_viz_dir = "./visualization_output"
experiment_name = "my_experiment"
html_file = MCTS.save_final_visualization(web_viz_dir, experiment_name)
print(f"Comprehensive visualization: {html_file}")
```

## Implementation Details

### Node and Edge Management
- **Unique IDs**: Each edge gets a unique ID (`e_{from}_{to}_{counter}`) for precise updates
- **Hidden Property**: Nodes and edges include a `hidden` boolean property
- **JSON Storage**: Complete tree data is stored in JSON format for comprehensive visualizations

### JavaScript Integration
- **PyVis Integration**: Injects JavaScript into PyVis-generated HTML files
- **Event Handling**: Uses vis.js `doubleClick` event for interaction
- **DataSet Updates**: Uses vis.js DataSet update methods for efficient rendering
- **Parent-Child Mapping**: Builds runtime maps of parent-child relationships

### Comprehensive HTML Features
- **Multi-Trial Support**: Navigate between different trials and steps
- **Consistent Behavior**: Same double-click functionality across all views
- **Event Management**: Properly handles event listeners when switching between snapshots

## Browser Compatibility
- **Modern Browsers**: Works with Chrome, Firefox, Safari, Edge
- **JavaScript Required**: Requires JavaScript to be enabled
- **Local Files**: Can be opened as local HTML files or served via web server

## Troubleshooting

### Common Issues
1. **No Double-Click Response**: Ensure `tree_visualization: True` in args
2. **All Nodes Visible**: Check `initial_visible_depth` setting
3. **Missing Children**: Verify tree has multiple levels from MCTS search

### Debugging
- **Console Logs**: Check browser console for JavaScript errors
- **JSON Data**: Inspect the `*_data.json` file for correct node/edge structure
- **Network Structure**: Verify edges have proper `from`, `to`, and `id` fields

## Performance Considerations
- **Large Trees**: For trees with 1000+ nodes, consider higher `initial_visible_depth`
- **Recursive Mode**: Use sparingly on very deep trees to avoid performance issues
- **Browser Memory**: Very large visualizations may require browser refresh

## Files Modified
- `src/algos/mcts.py`: Enhanced `tree_visualization()` and `save_comprehensive_html()`
- `tests/test_pyvis/test_double_click_expand.py`: Test and example implementation
- `tests/test_pyvis/test_compact_layout.py`: Test for compact layout functionality

## Testing

### Double-Click Functionality Test
```bash
cd tests/test_pyvis
python test_double_click_expand.py
```

### Compact Layout Test
```bash
cd tests/test_pyvis
python test_compact_layout.py
```

Both tests will generate HTML files that you can open in a browser to verify the functionality works correctly.
Run the test script to verify functionality:
```bash
python tests/test_pyvis/test_double_click_expand.py
```

The test creates sample visualizations and verifies:
- ✓ Double-click JavaScript injection
- ✓ Hidden property on nodes and edges
- ✓ Recursive mode functionality
- ✓ Comprehensive HTML generation
