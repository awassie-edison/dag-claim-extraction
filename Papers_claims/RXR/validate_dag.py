import json

with open("/Users/ozwassie/Desktop/Datasets/CAG Repeat Claims/260317_Testing_DAG_Claim_Extraction_V2/Papers_claims/RXR/claim_dag.json") as f:
    dag = json.load(f)

nodes = {n['id']: n for n in dag['nodes']}
edges = dag['edges']

roots = [n for n in dag['nodes'] if n['node_type'] == 'root']
intermediates = [n for n in dag['nodes'] if n['node_type'] == 'intermediate']
leaves = [n for n in dag['nodes'] if n['node_type'] == 'leaf']
lit_leaves = [n for n in dag['nodes'] if n.get('source') == 'literature']

print('=== DAG VALIDATION ===')
print()

leaf_ids = sorted([n['id'] for n in leaves])
expected_leaves = [f'L{i}' for i in range(1, 35)]
missing = set(expected_leaves) - set(leaf_ids)
extra = set(leaf_ids) - set(expected_leaves)
print(f'Leaf claims: {len(leaves)} (expected 34)')
if missing:
    print(f'  MISSING: {missing}')
if extra:
    print(f'  EXTRA: {extra}')
if not missing and not extra:
    print('  All 34 leaf claims present. PASS')

children_with_parents = set(e['child'] for e in edges)
non_roots = [n['id'] for n in dag['nodes'] if n['node_type'] != 'root']
orphans = [nid for nid in non_roots if nid not in children_with_parents]
print(f'Orphan check: {len(orphans)} orphans')
if orphans:
    print(f'  ORPHANS: {orphans}')
else:
    print('  No orphans. PASS')

adj = {}
for e in edges:
    adj.setdefault(e['child'], []).append(e['parent'])

def has_cycle():
    visited = set()
    rec_stack = set()
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for parent in adj.get(node, []):
            if parent not in visited:
                if dfs(parent):
                    return True
            elif parent in rec_stack:
                return True
        rec_stack.discard(node)
        return False
    for node in nodes:
        if node not in visited:
            if dfs(node):
                return True
    return False

cycle = has_cycle()
print(f'Cycle check: {"CYCLE DETECTED - FAIL" if cycle else "No cycles. PASS"}')

roots_with_parents = [r['id'] for r in roots if r['id'] in children_with_parents]
if roots_with_parents:
    print(f'Root validity: FAIL - roots have parents: {roots_with_parents}')
else:
    print('Root validity: All roots have zero parent edges. PASS')

parents_with_children = set(e['parent'] for e in edges)
non_leaves = [n['id'] for n in dag['nodes'] if n['node_type'] != 'leaf']
childless = [nid for nid in non_leaves if nid not in parents_with_children]
if childless:
    print(f'Non-leaf children check: FAIL - childless: {childless}')
else:
    print('Non-leaf children check: All non-leaf nodes have children. PASS')

computed_depth = {}
for r in roots:
    computed_depth[r['id']] = 0

changed = True
while changed:
    changed = False
    for e in edges:
        child, parent = e['child'], e['parent']
        if parent in computed_depth:
            new_depth = computed_depth[parent] + 1
            if child not in computed_depth or new_depth < computed_depth[child]:
                computed_depth[child] = new_depth
                changed = True

depth_errors = []
for n in dag['nodes']:
    if n['id'] in computed_depth and n['depth'] != computed_depth[n['id']]:
        depth_errors.append(f"{n['id']}: stated={n['depth']}, computed={computed_depth[n['id']]}")
if depth_errors:
    print(f'Depth validation: FAIL - {depth_errors}')
else:
    print('Depth validation: All depths correct. PASS')

max_depth = max(n['depth'] for n in dag['nodes'])
all_datasets = set()
for n in dag['nodes']:
    if n.get('dataset_accessions'):
        for acc in n['dataset_accessions']:
            if acc:
                all_datasets.add(acc)

print()
print('=== DAG SUMMARY ===')
print(f'Root nodes:         {len(roots)}')
print(f'Intermediate nodes: {len(intermediates)}')
print(f'Leaf nodes:         {len(leaves)} (dataset-derived)')
print(f'Literature leaves:  {len(lit_leaves)}')
print(f'Total nodes:        {len(dag["nodes"])}')
print(f'Total edges:        {len(edges)}')
print(f'Max depth:          {max_depth}')
print(f'Datasets:           {len(all_datasets)}')
for d in sorted(all_datasets):
    print(f'  - {d}')
print(f'Glossary terms:     {len(dag["glossary"])}')
print(f'Excluded claims:    {len(dag["excluded_borderline_claims"])}')
