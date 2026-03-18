#!/usr/bin/env python3
"""
Generate an interactive HTML visualization of a claim DAG.

Usage:
    python3 build_dag_visualization.py claim_dag.json [output.html]

If output.html is not specified, it defaults to claim_dag_viz.html in the
same directory as the input file.

The generated HTML is fully self-contained (JS libraries embedded inline)
and works offline.
"""

import json
import sys
import os
import urllib.request
import html as html_module

# --- JS library URLs (will be downloaded and embedded) ---
JS_LIBS = {
    "cytoscape": "https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js",
    "dagre": "https://unpkg.com/dagre@0.8.5/dist/dagre.min.js",
    "cytoscape_dagre": "https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js",
}

# --- Claim type color palette ---
CLAIM_TYPE_COLORS = {
    "direct empirical observation": "#4A90D9",
    "statistical/quantitative finding": "#2EAF6E",
    "threshold-based or cutoff-dependent result": "#E8913A",
    "computational/modeling-derived conclusion": "#9B59B6",
    "comparative or integrative finding": "#1ABC9C",
    "mechanistic interpretation": "#E74C3C",
}
DEFAULT_COLOR = "#95A5A6"

# --- Node type shapes ---
NODE_TYPE_SHAPES = {
    "root": "diamond",
    "intermediate": "round-rectangle",
    "leaf": "ellipse",
}


def fetch_js_library(url):
    """Download a JS library. Returns the JS source code as a string."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  WARNING: Could not download {url}: {e}")
        return None


def build_cytoscape_elements(dag):
    """Convert DAG JSON to Cytoscape.js elements format."""
    elements = []

    for node in dag["nodes"]:
        color = CLAIM_TYPE_COLORS.get(node.get("claim_type", ""), DEFAULT_COLOR)
        shape = NODE_TYPE_SHAPES.get(node.get("node_type", "leaf"), "ellipse")

        # Build label: short version for display
        claim_text = node.get("claim_text", "")
        label = claim_text[:60] + "..." if len(claim_text) > 60 else claim_text

        # Size based on node type
        if node["node_type"] == "root":
            width, height = 220, 60
        elif node["node_type"] == "intermediate":
            width, height = 180, 50
        else:
            width, height = 150, 40

        elements.append({
            "group": "nodes",
            "data": {
                "id": node["id"],
                "label": label,
                "claim_text": claim_text,
                "claim_type": node.get("claim_type", ""),
                "node_type": node.get("node_type", ""),
                "depth": node.get("depth", ""),
                "source": node.get("source", ""),
                "datasets": "; ".join(node.get("datasets") or []),
                "dataset_accessions": "; ".join(
                    str(a) for a in (node.get("dataset_accessions") or []) if a
                ),
                "experiment_type": node.get("experiment_type") or "",
                "experiment_detail": node.get("experiment_detail") or "",
                "analysis_type": node.get("analysis_type") or "",
                "analysis_detail": node.get("analysis_detail") or "",
                "evidence_location": node.get("evidence_location", ""),
                "supporting_quote": node.get("supporting_quote") or "",
                "dataset_link": node.get("dataset_link") or "",
                "color": color,
                "shape": shape,
                "width": width,
                "height": height,
            },
        })

    # Edge line styles by relationship type
    edge_styles = {
        "supports": "solid",
        "quantifies": "dashed",
        "qualifies": "dotted",
        "contrasts": "dashed",
        "interprets": "dashed",
        "enables": "dotted",
    }
    edge_colors = {
        "supports": "#7f8c8d",
        "quantifies": "#2EAF6E",
        "qualifies": "#E8913A",
        "contrasts": "#E74C3C",
        "interprets": "#9B59B6",
        "enables": "#3498DB",
    }

    for i, edge in enumerate(dag["edges"]):
        rel = edge.get("relationship", "supports")
        elements.append({
            "group": "edges",
            "data": {
                "id": f"e{i}",
                "source": edge["child"],
                "target": edge["parent"],
                "relationship": rel,
                "line_style": edge_styles.get(rel, "solid"),
                "line_color": edge_colors.get(rel, "#7f8c8d"),
            },
        })

    return elements


def generate_html(dag, elements_json, js_libs):
    """Generate the complete standalone HTML."""
    paper_title = dag.get("paper", {}).get("title", "Claim DAG")
    paper_year = dag.get("paper", {}).get("year", "")

    # Build legend HTML for claim types
    legend_items = ""
    for ct, color in CLAIM_TYPE_COLORS.items():
        legend_items += f'<div class="legend-item"><span class="legend-dot" style="background:{color}"></span>{html_module.escape(ct)}</div>\n'

    # Build legend for node types
    node_legend = ""
    shape_labels = {"root": "Root (thesis)", "intermediate": "Intermediate", "leaf": "Leaf (atomic claim)"}
    shape_symbols = {"root": "&#9670;", "intermediate": "&#9645;", "leaf": "&#9679;"}
    for nt, label in shape_labels.items():
        node_legend += f'<div class="legend-item"><span class="legend-shape">{shape_symbols[nt]}</span>{label}</div>\n'

    # Build legend for edge types
    edge_legend_items = ""
    edge_labels = {
        "supports": ("solid", "#7f8c8d"),
        "quantifies": ("dashed", "#2EAF6E"),
        "qualifies": ("dotted", "#E8913A"),
        "contrasts": ("dashed", "#E74C3C"),
        "interprets": ("dashed", "#9B59B6"),
        "enables": ("dotted", "#3498DB"),
    }
    for rel, (style, color) in edge_labels.items():
        dash = "none" if style == "solid" else ("4,4" if style == "dashed" else "2,2")
        edge_legend_items += f'''<div class="legend-item">
            <svg width="30" height="10"><line x1="0" y1="5" x2="30" y2="5"
                stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/></svg>
            {rel}</div>\n'''

    # Stats
    nodes = dag["nodes"]
    n_roots = sum(1 for n in nodes if n["node_type"] == "root")
    n_inter = sum(1 for n in nodes if n["node_type"] == "intermediate")
    n_leaves = sum(1 for n in nodes if n["node_type"] == "leaf")
    n_edges = len(dag["edges"])
    max_depth = max((n.get("depth", 0) for n in nodes), default=0)

    # Embed JS libraries
    cytoscape_js = js_libs.get("cytoscape", "")
    dagre_js = js_libs.get("dagre", "")
    cytoscape_dagre_js = js_libs.get("cytoscape_dagre", "")

    # If any library failed to download, use CDN fallback
    cytoscape_tag = f"<script>{cytoscape_js}</script>" if cytoscape_js else '<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>'
    dagre_tag = f"<script>{dagre_js}</script>" if dagre_js else '<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>'
    cytoscape_dagre_tag = f"<script>{cytoscape_dagre_js}</script>" if cytoscape_dagre_js else '<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(paper_title)} - Claim DAG</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; color: #2c3e50; }}

#header {{
    background: #2c3e50; color: white; padding: 12px 20px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
}}
#header h1 {{ font-size: 16px; font-weight: 600; flex: 1; min-width: 200px; }}
#search-box {{
    padding: 6px 12px; border: none; border-radius: 4px;
    font-size: 13px; width: 250px; outline: none;
}}
#search-box::placeholder {{ color: #95a5a6; }}
#stats {{ font-size: 12px; color: #bdc3c7; white-space: nowrap; }}
#btn-reset {{
    padding: 6px 12px; border: none; border-radius: 4px;
    background: #3498db; color: white; cursor: pointer; font-size: 12px;
}}
#btn-reset:hover {{ background: #2980b9; }}

#main {{ display: flex; height: calc(100vh - 48px); }}

#cy-container {{ flex: 1; position: relative; }}
#cy {{ width: 100%; height: 100%; }}

#sidebar {{
    width: 360px; background: white; border-left: 1px solid #ddd;
    overflow-y: auto; display: flex; flex-direction: column;
}}
#detail-panel {{ padding: 16px; flex: 1; }}
#detail-panel.empty {{ display: flex; align-items: center; justify-content: center; color: #95a5a6; font-size: 14px; }}
#detail-panel h2 {{ font-size: 14px; color: #7f8c8d; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
#claim-id {{ font-size: 13px; color: #3498db; font-weight: 600; margin-bottom: 4px; }}
#claim-text {{ font-size: 15px; line-height: 1.5; margin-bottom: 16px; color: #2c3e50; }}
.detail-row {{ margin-bottom: 8px; }}
.detail-label {{ font-size: 11px; color: #95a5a6; text-transform: uppercase; letter-spacing: 0.5px; }}
.detail-value {{ font-size: 13px; color: #2c3e50; margin-top: 2px; }}
.detail-long {{ font-size: 12px; line-height: 1.5; color: #34495e; background: #f8f9fa; padding: 8px 10px; border-radius: 4px; border-left: 3px solid #3498db; margin-top: 4px; }}
.detail-quote {{ font-size: 12px; line-height: 1.5; color: #555; font-style: italic; background: #fdf6e3; padding: 8px 10px; border-radius: 4px; border-left: 3px solid #e8913a; margin-top: 4px; }}
.detail-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 11px; font-weight: 600; color: white;
}}
#connections {{ margin-top: 16px; border-top: 1px solid #eee; padding-top: 12px; }}
#connections h3 {{ font-size: 12px; color: #7f8c8d; margin-bottom: 6px; }}
.conn-item {{
    font-size: 12px; padding: 4px 0; cursor: pointer; color: #3498db;
}}
.conn-item:hover {{ text-decoration: underline; }}
.conn-rel {{ color: #95a5a6; font-style: italic; }}

#legend-panel {{
    padding: 12px 16px; border-top: 1px solid #eee; background: #fafbfc;
}}
#legend-panel h3 {{ font-size: 11px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.legend-section {{ margin-bottom: 8px; }}
.legend-item {{ font-size: 11px; display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.legend-shape {{ font-size: 14px; width: 16px; text-align: center; flex-shrink: 0; }}
</style>
</head>
<body>

<div id="header">
    <h1>{html_module.escape(paper_title)} ({html_module.escape(paper_year)})</h1>
    <input type="text" id="search-box" placeholder="Search claims..." />
    <span id="stats">{n_roots}R / {n_inter}I / {n_leaves}L | {n_edges} edges | depth 0-{max_depth}</span>
    <button id="btn-reset">Reset View</button>
</div>

<div id="main">
    <div id="cy-container">
        <div id="cy"></div>
    </div>
    <div id="sidebar">
        <div id="detail-panel" class="empty">Click a node to view details</div>
        <div id="legend-panel">
            <div class="legend-section">
                <h3>Claim Type (color)</h3>
                {legend_items}
            </div>
            <div class="legend-section">
                <h3>Node Type (shape)</h3>
                {node_legend}
            </div>
            <div class="legend-section">
                <h3>Edge Type (line)</h3>
                {edge_legend_items}
            </div>
        </div>
    </div>
</div>

{cytoscape_tag}
{dagre_tag}
{cytoscape_dagre_tag}

<script>
const elements = {elements_json};

const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: elements,
    style: [
        {{
            selector: 'node',
            style: {{
                'label': 'data(label)',
                'text-wrap': 'wrap',
                'text-max-width': 'data(width)',
                'font-size': 10,
                'text-valign': 'center',
                'text-halign': 'center',
                'background-color': 'data(color)',
                'shape': 'data(shape)',
                'width': 'data(width)',
                'height': 'data(height)',
                'border-width': 2,
                'border-color': 'data(color)',
                'border-opacity': 0.3,
                'color': '#fff',
                'text-outline-width': 1.5,
                'text-outline-color': 'data(color)',
                'padding': '8px',
                'min-zoomed-font-size': 6,
            }}
        }},
        {{
            selector: 'node[node_type="root"]',
            style: {{
                'font-size': 12,
                'font-weight': 'bold',
                'border-width': 3,
                'border-opacity': 0.6,
            }}
        }},
        {{
            selector: 'edge',
            style: {{
                'width': 1.5,
                'line-color': 'data(line_color)',
                'target-arrow-color': 'data(line_color)',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 0.8,
                'line-style': 'data(line_style)',
                'opacity': 0.6,
            }}
        }},
        {{
            selector: '.highlighted',
            style: {{
                'border-width': 4,
                'border-color': '#f39c12',
                'border-opacity': 1,
                'z-index': 10,
            }}
        }},
        {{
            selector: '.highlighted-edge',
            style: {{
                'width': 3,
                'opacity': 1,
                'z-index': 10,
            }}
        }},
        {{
            selector: '.dimmed',
            style: {{
                'opacity': 0.15,
            }}
        }},
        {{
            selector: '.search-match',
            style: {{
                'border-width': 4,
                'border-color': '#f1c40f',
                'border-opacity': 1,
            }}
        }},
    ],
    layout: {{
        name: 'dagre',
        rankDir: 'TB',
        rankSep: 90,
        nodeSep: 30,
        edgeSep: 15,
        padding: 30,
    }},
    wheelSensitivity: 0.3,
    minZoom: 0.1,
    maxZoom: 3,
}});

// --- Click: show detail and highlight ancestors/descendants ---
cy.on('tap', 'node', function(evt) {{
    const node = evt.target;
    showDetail(node);
    highlightLineage(node);
}});

cy.on('tap', function(evt) {{
    if (evt.target === cy) {{
        clearHighlight();
        clearDetail();
    }}
}});

function showDetail(node) {{
    const d = node.data();
    const panel = document.getElementById('detail-panel');
    panel.className = '';

    // Find parent and child connections
    const incomingEdges = node.connectedEdges().filter(e => e.data('target') === d.id);
    const outgoingEdges = node.connectedEdges().filter(e => e.data('source') === d.id);

    let parentsHtml = '';
    outgoingEdges.forEach(e => {{
        const parentNode = cy.getElementById(e.data('target'));
        const pd = parentNode.data();
        parentsHtml += `<div class="conn-item" data-node-id="${{pd.id}}">`
            + `<span class="conn-rel">${{e.data('relationship')}}</span> `
            + `${{pd.id}}: ${{pd.claim_text.substring(0, 80)}}...</div>`;
    }});

    let childrenHtml = '';
    incomingEdges.forEach(e => {{
        const childNode = cy.getElementById(e.data('source'));
        const cd = childNode.data();
        childrenHtml += `<div class="conn-item" data-node-id="${{cd.id}}">`
            + `<span class="conn-rel">${{e.data('relationship')}}</span> `
            + `${{cd.id}}: ${{cd.claim_text.substring(0, 80)}}...</div>`;
    }});

    panel.innerHTML = `
        <h2>Node Detail</h2>
        <div id="claim-id">${{d.id}} <span class="detail-badge" style="background:${{d.color}}">${{d.node_type}}</span></div>
        <div id="claim-text">${{d.claim_text}}</div>

        ${{d.supporting_quote ? `<div class="detail-row">
            <div class="detail-label">Supporting Quote</div>
            <div class="detail-value detail-quote">"${{d.supporting_quote}}"</div>
        </div>` : ''}}

        <div class="detail-row">
            <div class="detail-label">Claim Type</div>
            <div class="detail-value">${{d.claim_type}}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Depth</div>
            <div class="detail-value">${{d.depth}}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Source</div>
            <div class="detail-value">${{d.source}}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Datasets</div>
            <div class="detail-value">${{d.datasets || 'N/A'}}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Accessions</div>
            <div class="detail-value">${{d.dataset_accessions || 'N/A'}}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Experiment Type</div>
            <div class="detail-value">${{d.experiment_type || 'N/A'}}</div>
        </div>
        ${{d.experiment_detail ? `<div class="detail-row">
            <div class="detail-label">Experiment Detail</div>
            <div class="detail-value detail-long">${{d.experiment_detail}}</div>
        </div>` : ''}}
        <div class="detail-row">
            <div class="detail-label">Analysis Type</div>
            <div class="detail-value">${{d.analysis_type || 'N/A'}}</div>
        </div>
        ${{d.analysis_detail ? `<div class="detail-row">
            <div class="detail-label">Analysis Detail</div>
            <div class="detail-value detail-long">${{d.analysis_detail}}</div>
        </div>` : ''}}
        <div class="detail-row">
            <div class="detail-label">Evidence Location</div>
            <div class="detail-value">${{d.evidence_location}}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Dataset Link</div>
            <div class="detail-value">${{d.dataset_link || 'N/A'}}</div>
        </div>

        <div id="connections">
            ${{parentsHtml ? '<h3>Parents (this claim supports):</h3>' + parentsHtml : ''}}
            ${{childrenHtml ? '<h3>Children (support this claim):</h3>' + childrenHtml : ''}}
        </div>
    `;

    // Make connection items clickable
    panel.querySelectorAll('.conn-item').forEach(item => {{
        item.addEventListener('click', () => {{
            const nodeId = item.getAttribute('data-node-id');
            const targetNode = cy.getElementById(nodeId);
            showDetail(targetNode);
            highlightLineage(targetNode);
            cy.animate({{ center: {{ eles: targetNode }}, duration: 300 }});
        }});
    }});
}}

function clearDetail() {{
    const panel = document.getElementById('detail-panel');
    panel.className = 'empty';
    panel.innerHTML = 'Click a node to view details';
}}

function highlightLineage(node) {{
    // Clear previous
    cy.elements().removeClass('highlighted highlighted-edge dimmed');

    // Find all ancestors (successors in our edge direction: child->parent means edges go source->target)
    const ancestors = node.successors();
    // Find all descendants (predecessors)
    const descendants = node.predecessors();

    const lineage = ancestors.union(descendants).union(node);
    const lineageNodes = lineage.nodes();
    const lineageEdges = lineage.edges();

    // Dim everything
    cy.elements().addClass('dimmed');

    // Highlight lineage
    lineageNodes.removeClass('dimmed').addClass('highlighted');
    lineageEdges.removeClass('dimmed').addClass('highlighted-edge');
    node.removeClass('dimmed').addClass('highlighted');
}}

function clearHighlight() {{
    cy.elements().removeClass('highlighted highlighted-edge dimmed search-match');
}}

// --- Search ---
const searchBox = document.getElementById('search-box');
let searchTimeout;
searchBox.addEventListener('input', function() {{
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {{
        const query = searchBox.value.trim().toLowerCase();
        cy.elements().removeClass('dimmed search-match');

        if (!query) return;

        const matchingNodes = cy.nodes().filter(n => {{
            const d = n.data();
            return d.claim_text.toLowerCase().includes(query)
                || d.id.toLowerCase().includes(query)
                || (d.claim_type && d.claim_type.toLowerCase().includes(query))
                || (d.evidence_location && d.evidence_location.toLowerCase().includes(query))
                || (d.supporting_quote && d.supporting_quote.toLowerCase().includes(query))
                || (d.experiment_detail && d.experiment_detail.toLowerCase().includes(query))
                || (d.analysis_detail && d.analysis_detail.toLowerCase().includes(query));
        }});

        if (matchingNodes.length > 0) {{
            cy.elements().addClass('dimmed');
            matchingNodes.removeClass('dimmed').addClass('search-match');
            matchingNodes.connectedEdges().removeClass('dimmed');
        }}
    }}, 200);
}});

// --- Reset button ---
document.getElementById('btn-reset').addEventListener('click', function() {{
    clearHighlight();
    clearDetail();
    searchBox.value = '';
    cy.fit(undefined, 30);
}});

// --- Keyboard shortcut: Escape to reset ---
document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') {{
        clearHighlight();
        clearDetail();
        searchBox.value = '';
    }}
}});

// Initial fit
cy.fit(undefined, 30);
</script>

</body>
</html>"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 build_dag_visualization.py claim_dag.json [output.html]")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.join(
            os.path.dirname(input_path),
            os.path.splitext(os.path.basename(input_path))[0] + "_viz.html",
        )

    print(f"Reading DAG from: {input_path}")
    with open(input_path) as f:
        dag = json.load(f)

    print(f"  Nodes: {len(dag['nodes'])}, Edges: {len(dag['edges'])}")

    # Download JS libraries for embedding
    print("Downloading JS libraries for offline embedding...")
    js_libs = {}
    for name, url in JS_LIBS.items():
        print(f"  Fetching {name}...")
        js_libs[name] = fetch_js_library(url)

    # Build Cytoscape elements
    elements = build_cytoscape_elements(dag)
    elements_json = json.dumps(elements)

    # Generate HTML
    print("Generating HTML...")
    html_content = generate_html(dag, elements_json, js_libs)

    # Write output
    with open(output_path, "w") as f:
        f.write(html_content)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Written to: {output_path} ({size_kb:.0f} KB)")
    print("Open in a browser to view the interactive DAG.")


if __name__ == "__main__":
    main()
