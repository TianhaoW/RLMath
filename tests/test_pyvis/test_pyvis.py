# build_multistate_html.py
import json
from pyvis.network import Network

def make_net_state(i):
    net = Network(height="600px", width="100%", directed=True)
    # 这里演示每次结构略有不同；你可以替换成你自己的生成逻辑
    net.add_node(1, label=f"State {i} - Node 1")
    net.add_node(2, label=f"State {i} - Node 2")
    if i % 2 == 0:
        net.add_node(3, label=f"State {i} - Node 3")
        net.add_edge(1, 2)
        net.add_edge(2, 3)
    else:
        net.add_edge(1, 2)
        net.add_edge(1, 1)  # self-loop 示例
    # 关键：PyVis 的 net.nodes / net.edges 就是可序列化的列表
    return {"id": f"v{i}", "title": f"Snapshot {i}", "nodes": net.nodes, "edges": net.edges}

snapshots = [make_net_state(i) for i in range(5)]  # 假设有 5 个版本

HTML = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>PyVis Multi-Snapshot Viewer</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }}
    #toolbar {{ display:flex; gap:8px; align-items:center; padding:10px; border-bottom:1px solid #eee; }}
    #mynetwork {{ height: calc(100vh - 54px); }}
    button, select {{ padding:6px 10px; border-radius:8px; border:1px solid #ddd; background:#fafafa; }}
    #title {{ font-weight:600; margin-left:8px; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="prev">⟵ Prev</button>
    <button id="next">Next ⟶</button>
    <select id="jump"></select>
    <span id="title"></span>
  </div>
  <div id="mynetwork"></div>

  <!-- vis-network (vis.js) -->
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <script>
    const snapshots = {json.dumps(snapshots, ensure_ascii=False)};
    let idx = 0;

    // 初始化 DataSet 与 Network
    const nodes = new vis.DataSet([]);
    const edges = new vis.DataSet([]);
    const container = document.getElementById('mynetwork');
    const network = new vis.Network(container, {{ nodes, edges }}, {{
      physics: true,
      interaction: {{ hover: true }},
      layout: {{ improvedLayout: true }}
    }});

    // 切换加载某个快照
    function load(i) {{
      idx = (i + snapshots.length) % snapshots.length;
      const s = snapshots[idx];
      nodes.clear(); edges.clear();
      nodes.add(s.nodes); edges.add(s.edges);
      document.getElementById('title').textContent = s.title + " (" + s.id + ")";
      document.getElementById('jump').value = String(idx);
      // 适当缩放到合适视图
      requestAnimationFrame(() => network.fit({{ animation: true }}));
    }}

    // UI 绑定
    document.getElementById('prev').onclick = () => load(idx - 1);
    document.getElementById('next').onclick = () => load(idx + 1);

    const sel = document.getElementById('jump');
    snapshots.forEach((s, i) => {{
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = s.title;
      sel.appendChild(opt);
    }});
    sel.onchange = (e) => load(parseInt(e.target.value));

    // 启动
    load(0);
  </script>
</body>
</html>
"""

with open("multi_state_net.html", "w", encoding="utf-8") as f:
    f.write(HTML)

print("Wrote multi_state_net.html")