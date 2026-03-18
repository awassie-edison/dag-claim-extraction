import json
from collections import deque, Counter

with open("claim_dag.json") as f:
    dag = json.load(f)

nodes_map = {}
for n in dag["nodes"]:
    nodes_map[n["id"]] = n
edges = dag["edges"]

roots = [n for n in dag["nodes"] if n["node_type"] == "root"]
intermediates = [n for n in dag["nodes"] if n["node_type"] == "intermediate"]
leaves_dataset = [n for n in dag["nodes"] if n["node_type"] == "leaf" and n["source"] == "dataset"]
leaves_lit = [n for n in dag["nodes"] if n["node_type"] == "leaf" and n["source"] == "literature"]

print("=== DAG SUMMARY ===")
print("Root nodes:", len(roots))
print("Intermediate nodes:", len(intermediates))
print("Dataset-derived leaf nodes:", len(leaves_dataset))
print("Literature-derived leaf nodes:", len(leaves_lit))
print("Total nodes:", len(dag["nodes"]))
print("Total edges:", len(edges))
print()

# Check all 38 leaf claims present
leaf_ids = sorted([n["id"] for n in leaves_dataset])
expected = ["L" + str(i) for i in range(1, 39)]
missing = set(expected) - set(leaf_ids)
extra = set(leaf_ids) - set(expected)
print("=== LEAF COMPLETENESS ===")
print("Expected leaf IDs: L1-L38")
print("Missing leaves:", missing if missing else "NONE")
print("Extra leaves:", extra if extra else "NONE")
print()

# Build parent/child maps
parent_map = {}
child_map = {}
for e in edges:
    if e["child"] not in parent_map:
        parent_map[e["child"]] = []
    parent_map[e["child"]].append(e["parent"])
    if e["parent"] not in child_map:
        child_map[e["parent"]] = []
    child_map[e["parent"]].append(e["child"])

# Check roots have zero parents
print("=== ROOT VALIDATION ===")
for r in roots:
    if r["id"] in parent_map:
        print("ERROR: Root", r["id"], "has parents:", parent_map[r["id"]])
    else:
        print("Root", r["id"], ": zero parent edges OK")

# Check no orphans
all_ids = set(nodes_map.keys())
root_ids = set(r["id"] for r in roots)
non_root_ids = all_ids - root_ids
orphans = non_root_ids - set(parent_map.keys())
print("Orphan non-root nodes:", orphans if orphans else "NONE")
print()

# Check intermediates have children
print("=== INTERMEDIATE VALIDATION ===")
ok_count = 0
for n in intermediates:
    if n["id"] not in child_map:
        print("ERROR: Intermediate", n["id"], "has no children")
    else:
        ok_count += 1
print("Intermediates with children:", ok_count, "/", len(intermediates))
print()

# Acyclicity check via topological sort
print("=== ACYCLICITY CHECK ===")
in_degree = {}
adj = {}
for nid in all_ids:
    in_degree[nid] = 0
    adj[nid] = []
for e in edges:
    adj[e["child"]].append(e["parent"])
    in_degree[e["parent"]] += 1

queue = deque()
for nid in all_ids:
    if in_degree[nid] == 0:
        queue.append(nid)
visited = 0
while queue:
    curr = queue.popleft()
    visited += 1
    for neighbor in adj[curr]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

if visited == len(all_ids):
    print("PASSED (no cycles)")
else:
    print("CYCLE DETECTED: only", visited, "/", len(all_ids), "nodes in topological order")
print()

# Depth validation
print("=== DEPTH VALIDATION ===")
memo = {}
def compute_depth(nid):
    if nid in memo:
        return memo[nid]
    if nid not in parent_map:
        memo[nid] = 0
        return 0
    d = min(compute_depth(p) for p in parent_map[nid]) + 1
    memo[nid] = d
    return d

depth_errors = []
for n in dag["nodes"]:
    expected_d = compute_depth(n["id"])
    if n["depth"] != expected_d:
        depth_errors.append((n["id"], n["depth"], expected_d))

if depth_errors:
    print("Depth errors:", len(depth_errors))
    for nid, stated, computed in depth_errors:
        print("  ", nid, ": stated=", stated, ", computed=", computed)
else:
    print("ALL CORRECT")

max_depth = max(n["depth"] for n in dag["nodes"])
print("Max depth:", max_depth)
print()

# Datasets
print("=== DATASETS ===")
all_datasets = set()
for n in dag["nodes"]:
    ds = n.get("datasets")
    if ds and isinstance(ds, list):
        for d in ds:
            all_datasets.add(d)
print("Number of datasets:", len(all_datasets))
for d in all_datasets:
    print(" ", d)
print()

# Edge relationships
print("=== EDGE RELATIONSHIPS ===")
rel_counts = Counter(e["relationship"] for e in edges)
for rel, count in sorted(rel_counts.items()):
    print(" ", rel, ":", count)
