import os
import io
import json
import base64
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network

# Assume Node class is defined elsewhere

class MCTSVisualizer:
    # Global storage for all trials and steps (class variable)
    global_trial_data = []

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.snapshots = []  # Store snapshots for the current trial
        self.trial_id = 'unknown'

    def start_new_trial(self, trial_id):
        """Prepares the visualizer for a new trial."""
        self.trial_id = trial_id
        self.snapshots = [] # Clear snapshots from the previous trial

    def _state_to_image_base64(self, state):
        """
        Convert a game state to a base64-encoded image using the game's display_state method.
        """
        plt.figure(figsize=(4, 4))
        rows, cols = self.game.row_count, self.game.column_count
        y_idx, x_idx = np.nonzero(state)
        y_disp = rows - 1 - y_idx
        plt.scatter(x_idx, y_disp, s=200, c='blue', linewidths=0.5)
        plt.xticks(range(cols))
        plt.yticks(range(rows))
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.5, cols - 0.5)
        plt.ylim(-0.5, rows - 0.5)
        plt.gca().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, pad_inches=0.1)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        return f"data:image/png;base64,{img_base64}"

    def _get_node_label(self, node, iter_num=None):
        """
        Generate a label for a node showing its statistics.
        """
        avg_value = node.value_sum / node.visit_count if node.visit_count > 0 else 0
        ucb = 0
        if node.parent is not None and node.visit_count > 0:
            try:
                ucb = node.parent.get_ucb(node, iter_num or 0)
            except:
                ucb = 0 # Failsafe
        
        label = f"Visits: {node.visit_count}\n"
        label += f"Value Sum: {node.value_sum:.3f}\n"
        label += f"Avg Value: {avg_value:.3f}\n"
        label += f"UCB: {ucb:.3f}"
        return label

    def create_tree_snapshot(self, root, snapshot_name="MCTS Tree"):
        """
        Create a tree visualization snapshot for the current step.
        """
        # (This is your tree_visualization method, renamed for clarity)
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        net.barnes_hut()
        
        all_nodes, visited = [], set()
        def collect_nodes_dfs(node, level=0):
            if id(node) in visited: return
            visited.add(id(node))
            all_nodes.append((node, level))
            for child in node.children:
                collect_nodes_dfs(child, level + 1)
        
        collect_nodes_dfs(root)
        print(f"Tree visualization: Found {len(all_nodes)} nodes total")

        json_nodes, json_edges, node_mapping = [], [], {}
        for i, (node, level) in enumerate(all_nodes):
            current_id = f"node_{i}"
            node_mapping[id(node)] = current_id
            img_base64 = self._state_to_image_base64(node.state)
            label = self._get_node_label(node)
            
            color = "#4CAF50"  # Default green
            if node.is_fully_expanded(): color = "#2196F3"
            elif len(node.children) == 0 and not node.is_fully_expanded(): color = "#FF9800"
            elif np.sum(node.valid_moves) == 0: color = "#F44336"

            label_lines = label.split('\n')
            escaped_label = label.replace('\n', '\\n')
            title_text = f"Action: {node.action_taken}\\n{escaped_label}\\nChildren: {len(node.children)}\\nValid moves left: {np.sum(node.valid_moves)}"
            
            json_nodes.append({
                "id": current_id, "label": label_lines, "image": img_base64,
                "shape": "image", "size": 30, "level": level, "color": color,
                "title": title_text, "x": i * 100, "y": level * 150
            })
        
        for node, _ in all_nodes:
            current_id = node_mapping[id(node)]
            for child in node.children:
                if id(child) in node_mapping:
                    child_id = node_mapping[id(child)]
                    json_edges.append({
                        "from": current_id, "to": child_id,
                        "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4}
                    })
        
        snapshot_data = {
            'name': snapshot_name, 'trial_id': self.trial_id,
            'step_number': len(self.snapshots), 'total_nodes': len(all_nodes),
            'args': self.args.copy(), 'json_nodes': json_nodes, 'json_edges': json_edges
        }
        
        self.snapshots.append(snapshot_data)
        MCTSVisualizer.global_trial_data.append(snapshot_data)
        print(f"Snapshot created for Trial {self.trial_id}, Step {len(self.snapshots)}")

    @classmethod
    def clear_global_data(cls):
        """Clear all global trial data."""
        cls.global_trial_data.clear()
        print("Global trial data cleared.")

    @classmethod
    def save_final_visualization(cls, web_viz_dir=None, experiment_name="mcts_experiment"):
        """
        Save the final comprehensive visualization at the end of all trials.
        """
        # (This is your original save_final_visualization method)
        if not cls.global_trial_data:
            print("No global trial data to save.")
            return None
        if web_viz_dir is None:
            web_viz_dir = './web_visualization'
        os.makedirs(web_viz_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(web_viz_dir, f"{experiment_name}_comprehensive_{timestamp}.html")
        cls._save_comprehensive_html(filename)
        return filename

    @classmethod
    def _save_comprehensive_html(cls, filename="mcts_comprehensive_visualization.html"):
        """
        Creates the comprehensive HTML file with all trials and steps.
        """
        # (This is your original save_comprehensive_html method)
        if not cls.global_trial_data:
            print("No global trial data to save.")
            return

        json_snapshots = []
        for snapshot in cls.global_trial_data:
            json_snapshots.append({
                "id": f"t{snapshot['trial_id']}_s{snapshot['step_number']}",
                "title": f"Trial {snapshot['trial_id']} - {snapshot['name']}",
                "trial_id": snapshot['trial_id'],
                "step_number": snapshot['step_number'],
                "total_nodes": snapshot['total_nodes'],
                "nodes": snapshot.get('json_nodes', []),
                "edges": snapshot.get('json_edges', []),
                "grid_size": snapshot.get('args', {}).get('n', 'unknown')
            })
        
        # Save the JSON data separately for debugging and external use
        json_filename = filename.replace('.html', '_data.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_snapshots, f, indent=2, ensure_ascii=False)
        print(f"JSON data saved to: {json_filename}")

        # The large HTML string template goes here. It's omitted for brevity but is identical
        # to the one in your original code.
        html_content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>MCTS Comprehensive Tree Visualization</title>
  <style>
    /* ... Your CSS styles ... */
  </style>
</head>
<body>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <script>
    const snapshots = {json.dumps(json_snapshots, ensure_ascii=False, indent=2)};
    // ... The rest of your JavaScript for navigation, rendering, etc. ...
  </script>
</body>
</html>
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Comprehensive visualization saved to: {filename}")