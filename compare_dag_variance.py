#!/usr/bin/env python3
"""
Compare variance across repeated DAG claim extractions of the same paper.

Supports two modes:
  --claude    Use Claude Opus 4.6 for pairwise claim matching (recommended, requires ANTHROPIC_API_KEY)
  --embed     Use sentence-transformers cosine similarity (fallback, no API needed)

Outputs:
  - Console report with summary statistics
  - variance_report/  directory with PNG visualizations:
      structural_comparison.png      — per-run breakdown (2×2 grid)
      structural_variability.png     — mean ± range across runs
      claim_stability.png            — stacked bar: claims by repeat count & node type
      literature_claims_analysis.png — literature claim counts, stability, comparison
"""

import argparse
import json
import time
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PAPERS_DIR = PROJECT_ROOT / "Papers_claims"
OUTPUT_DIR = PROJECT_ROOT / "variance_report"
RUN_NAMES = sorted(
    [d.name for d in PAPERS_DIR.iterdir() if d.is_dir() and d.name.startswith("RXR_")]
)
# Cosine-similarity threshold for embedding-based matching
SIMILARITY_THRESHOLD = 0.82
# Embedding model (used only in --embed mode)
MODEL_NAME = "all-MiniLM-L6-v2"
# Claude model for semantic matching
CLAUDE_MODEL = "claude-opus-4-6"

# Plot style for presentation
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# 1. Load DAGs
# ---------------------------------------------------------------------------
def load_dags():
    """Load all DAG JSON files into a dict keyed by run name."""
    dags = {}
    for name in RUN_NAMES:
        path = PAPERS_DIR / name / "claim_dag.json"
        with open(path) as f:
            dags[name] = json.load(f)
    return dags


# ---------------------------------------------------------------------------
# 2. Structural summary
# ---------------------------------------------------------------------------
def structural_summary(dags):
    """Produce a DataFrame of structural metrics per run."""
    rows = []
    for name, dag in dags.items():
        nodes = dag["nodes"]
        edges = dag["edges"]
        excluded = dag.get("excluded_borderline_claims", [])
        depths = [n["depth"] for n in nodes]
        node_types = Counter(n["node_type"] for n in nodes)
        claim_types = Counter(n["claim_type"] for n in nodes)
        sources = Counter(n["source"] for n in nodes)
        rel_types = Counter(e["relationship"] for e in edges)
        row = {
            "run": name,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "roots": node_types.get("root", 0),
            "intermediates": node_types.get("intermediate", 0),
            "leaves": node_types.get("leaf", 0),
            "literature_nodes": sources.get("literature", 0),
            "dataset_nodes": sources.get("dataset", 0),
            "max_depth": max(depths),
            "mean_depth": round(np.mean(depths), 2),
            "n_excluded": len(excluded),
            "glossary_terms": len(dag.get("glossary", {})),
        }
        for ct in sorted(set(n["claim_type"] for d in dags.values() for n in d["nodes"])):
            row[f"ct_{ct}"] = claim_types.get(ct, 0)
        for rt in sorted(set(e["relationship"] for d in dags.values() for e in d["edges"])):
            row[f"rel_{rt}"] = rel_types.get(rt, 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Pairwise claim matching — Claude
# ---------------------------------------------------------------------------
def _parse_json_response(raw):
    """Robustly extract a JSON object from a Claude response string."""
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                raw = candidate
                break
    if raw.startswith("{"):
        depth = 0
        for i, ch in enumerate(raw):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = raw[: i + 1]
                    break
    return json.loads(raw)


def compare_pair_with_claude(client, run_a, claims_a, run_b, claims_b):
    """Send one pairwise comparison to Claude. Returns match result dict."""
    lines_a = [f"{i}|{c['id']}|{c['claim_text']}" for i, c in enumerate(claims_a)]
    lines_b = [f"{i}|{c['id']}|{c['claim_text']}" for i, c in enumerate(claims_b)]

    prompt = f"""You are an expert in biology and scientific claim analysis. Below are two lists of scientific claims extracted from the same paper by two independent runs of an extraction pipeline. Match claims between the two lists that describe the SAME scientific finding.

Two claims match if a domain expert would say they assert the same thing about the same biological phenomenon, regardless of wording differences.

Rules:
- Same gene/pathway/cell type + same directional assertion = match.
- Differing only in quantitative detail (e.g., "doubles" vs "increases significantly") = match.
- Different genes, tissues, or directions of change = NOT a match.
- A broad summary claim vs a specific sub-finding = NOT a match.
- Each claim matches at most ONE claim from the other list (best match).

RUN A ({run_a}) — {len(claims_a)} claims:
{chr(10).join(lines_a)}

RUN B ({run_b}) — {len(claims_b)} claims:
{chr(10).join(lines_b)}

Return a JSON object with:
- "matches": array of [A_INDEX, B_INDEX] pairs for matched claims
- "unmatched_a": array of A_INDEX values with no match in B
- "unmatched_b": array of B_INDEX values with no match in A

Return ONLY the JSON object, no other text."""

    t0 = time.time()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0

    parsed = _parse_json_response(response.content[0].text.strip())
    raw_matches = parsed.get("matches", [])

    # Validate: in-range and 1-to-1
    valid_matches = []
    used_a, used_b = set(), set()
    for pair in raw_matches:
        if len(pair) >= 2:
            a_idx, b_idx = int(pair[0]), int(pair[1])
            if (0 <= a_idx < len(claims_a) and 0 <= b_idx < len(claims_b)
                    and a_idx not in used_a and b_idx not in used_b):
                valid_matches.append((a_idx, b_idx))
                used_a.add(a_idx)
                used_b.add(b_idx)

    return {
        "run_a": run_a,
        "run_b": run_b,
        "matches": valid_matches,
        "n_claims_a": len(claims_a),
        "n_claims_b": len(claims_b),
        "elapsed": round(elapsed, 1),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def run_pairwise_claude(dags):
    """
    Run all pairwise comparisons with Claude. Returns per-run claim lists
    and the raw match results.
    """
    import anthropic
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = anthropic.Anthropic()

    # Build per-run claim lists
    run_claims = {}
    for name in sorted(dags):
        dag = dags[name]
        run_claims[name] = [
            {
                "id": node["id"],
                "claim_text": node["claim_text"],
                "claim_type": node["claim_type"],
                "node_type": node["node_type"],
                "depth": node["depth"],
                "source": node["source"],
            }
            for node in dag["nodes"]
        ]

    runs = sorted(run_claims)
    pairs = list(combinations(runs, 2))
    print(f"\nRunning {len(pairs)} pairwise comparisons with Claude {CLAUDE_MODEL} ...")

    all_results = []

    def do_compare(run_a, run_b):
        return compare_pair_with_claude(
            client, run_a, run_claims[run_a], run_b, run_claims[run_b]
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(do_compare, ra, rb): (ra, rb)
            for ra, rb in pairs
        }
        for future in as_completed(futures):
            result = future.result()
            n_m = len(result["matches"])
            print(f"  {result['run_a']} vs {result['run_b']}: "
                  f"{n_m} matches "
                  f"({result['n_claims_a']} vs {result['n_claims_b']} claims, "
                  f"{result['elapsed']}s)")
            all_results.append(result)

    total_matches = sum(len(r["matches"]) for r in all_results)
    total_tokens_in = sum(r["input_tokens"] for r in all_results)
    total_tokens_out = sum(r["output_tokens"] for r in all_results)
    print(f"\n  Total: {total_matches} matches across {len(pairs)} pairs")
    print(f"  Tokens: {total_tokens_in:,} input, {total_tokens_out:,} output")

    # Save audit trail
    audit = {
        "model": CLAUDE_MODEL,
        "method": "pairwise",
        "n_pairs": len(pairs),
        "total_matches": total_matches,
        "pair_results": [
            {
                "run_a": r["run_a"],
                "run_b": r["run_b"],
                "n_matches": len(r["matches"]),
                "matches": r["matches"],
                "elapsed": r["elapsed"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
            }
            for r in sorted(all_results, key=lambda r: (r["run_a"], r["run_b"]))
        ],
    }
    with open(OUTPUT_DIR / "claude_pairwise_results.json", "w") as f:
        json.dump(audit, f, indent=2)

    return run_claims, all_results


# ---------------------------------------------------------------------------
# 3b. Pairwise claim matching — embeddings (--embed fallback)
# ---------------------------------------------------------------------------
def run_pairwise_embed(dags):
    """
    Run all pairwise comparisons using cosine similarity of embeddings.
    Returns per-run claim lists and match results in the same format as Claude.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    # Build per-run claim lists and embed
    run_claims = {}
    run_embeddings = {}
    for name in sorted(dags):
        dag = dags[name]
        claims = [
            {
                "id": node["id"],
                "claim_text": node["claim_text"],
                "claim_type": node["claim_type"],
                "node_type": node["node_type"],
                "depth": node["depth"],
                "source": node["source"],
            }
            for node in dag["nodes"]
        ]
        run_claims[name] = claims
        texts = [c["claim_text"] for c in claims]
        print(f"  Embedding {name}: {len(texts)} claims ...")
        run_embeddings[name] = model.encode(texts, normalize_embeddings=True)

    runs = sorted(run_claims)
    pairs = list(combinations(runs, 2))
    print(f"\nRunning {len(pairs)} pairwise comparisons (threshold={SIMILARITY_THRESHOLD}) ...")

    all_results = []
    for run_a, run_b in pairs:
        sim = cosine_similarity(run_embeddings[run_a], run_embeddings[run_b])
        matches = []
        used_b = set()
        # Greedy best-match from A → B
        for a_idx in range(len(run_claims[run_a])):
            best_b = -1
            best_sim = SIMILARITY_THRESHOLD
            for b_idx in range(len(run_claims[run_b])):
                if b_idx not in used_b and sim[a_idx, b_idx] > best_sim:
                    best_sim = sim[a_idx, b_idx]
                    best_b = b_idx
            if best_b >= 0:
                matches.append((a_idx, best_b))
                used_b.add(best_b)

        print(f"  {run_a} vs {run_b}: {len(matches)} matches")
        all_results.append({
            "run_a": run_a,
            "run_b": run_b,
            "matches": matches,
            "n_claims_a": len(run_claims[run_a]),
            "n_claims_b": len(run_claims[run_b]),
        })

    return run_claims, all_results


# ---------------------------------------------------------------------------
# 4. Claim-level stability
# ---------------------------------------------------------------------------
def compute_claim_stability(run_claims, all_results):
    """
    For each claim in each run, count how many other runs contain a matching
    claim (based on the pairwise results).

    Returns a DataFrame with one row per claim and an 'n_runs' column (1–N)
    indicating how many runs (including its own) contain that claim.
    """
    # Build match-partner sets: (run, idx) → set of other runs it matched in
    match_partners = {}
    for name, claims in run_claims.items():
        for i in range(len(claims)):
            match_partners[(name, i)] = set()

    for result in all_results:
        for a_idx, b_idx in result["matches"]:
            match_partners[(result["run_a"], a_idx)].add(result["run_b"])
            match_partners[(result["run_b"], b_idx)].add(result["run_a"])

    # Build DataFrame
    records = []
    for name in sorted(run_claims):
        for i, claim in enumerate(run_claims[name]):
            records.append({
                "run": name,
                "id": claim["id"],
                "claim_text": claim["claim_text"],
                "claim_type": claim["claim_type"],
                "node_type": claim["node_type"],
                "depth": claim["depth"],
                "source": claim["source"],
                "n_runs": 1 + len(match_partners[(name, i)]),
            })

    return pd.DataFrame(records)


def print_claim_stability_summary(claims_df, n_runs):
    """Print claim-level stability distribution to console."""
    total = len(claims_df)
    print(f"\n=== Claim-Level Stability ({total} claims across {n_runs} runs) ===")
    for k in range(1, n_runs + 1):
        c = (claims_df["n_runs"] == k).sum()
        pct = 100 * c / total
        label = {1: "unique to 1 run", n_runs: "found in ALL runs"}.get(
            k, f"in {k} runs"
        )
        print(f"  {label}: {c:3d} ({pct:5.1f}%)")

    print("\nBy node type:")
    for nt in ["root", "intermediate", "leaf"]:
        subset = claims_df[claims_df["node_type"] == nt]
        if len(subset) == 0:
            continue
        all_runs = (subset["n_runs"] == n_runs).sum()
        print(f"  {nt:>14s}: {len(subset):3d} claims, "
              f"{all_runs} in all runs ({100 * all_runs / len(subset):.0f}%)")

    print("\nBy source:")
    for src in ["dataset", "literature"]:
        subset = claims_df[claims_df["source"] == src]
        if len(subset) == 0:
            continue
        all_runs = (subset["n_runs"] == n_runs).sum()
        print(f"  {src:>14s}: {len(subset):3d} claims, "
              f"{all_runs} in all runs ({100 * all_runs / len(subset):.0f}%)")

    # Per-run breakdown
    print("\nPer-run breakdown (claims found in all runs / total):")
    for run_name in sorted(claims_df["run"].unique()):
        run_df = claims_df[claims_df["run"] == run_name]
        stable = (run_df["n_runs"] == n_runs).sum()
        print(f"  {run_name}: {stable}/{len(run_df)} "
              f"({100 * stable / len(run_df):.0f}%) claims reproduced in all runs")


def print_literature_details(dags, n_runs):
    """Print detailed literature-claim information to console."""
    print("\n" + "=" * 70)
    print("LITERATURE-DERIVED CLAIMS — DETAILS")
    print("=" * 70)

    lit_counts = {}
    for name, dag in dags.items():
        lit_nodes = [n for n in dag["nodes"] if n["source"] == "literature"]
        lit_counts[name] = len(lit_nodes)

    print(f"\nCount per run:")
    for name in sorted(lit_counts):
        print(f"  {name}: {lit_counts[name]}")
    total = sum(lit_counts.values())
    print(f"  Mean: {total / n_runs:.1f}, Range: "
          f"{min(lit_counts.values())}–{max(lit_counts.values())}")

    print(f"\nDAG role of literature claims:")
    for name, dag in dags.items():
        lit_ids = {n["id"] for n in dag["nodes"] if n["source"] == "literature"}
        if not lit_ids:
            print(f"  {name}: no literature nodes")
            continue
        id_to_node = {n["id"]: n for n in dag["nodes"]}
        print(f"\n  {name}: {len(lit_ids)} literature node(s)")
        for lid in sorted(lit_ids):
            node = id_to_node[lid]
            parents = [e for e in dag["edges"] if e["child"] == lid]
            children = [e for e in dag["edges"] if e["parent"] == lid]
            print(f"    {lid}: \"{node['claim_text'][:70]}...\"")
            for p in parents:
                pnode = id_to_node[p["parent"]]
                print(f"      ↑ parent: {p['parent']} ({p['relationship']}) "
                      f"[{pnode['node_type']}]: \"{pnode['claim_text'][:60]}...\"")
            for c in children:
                cnode = id_to_node[c["child"]]
                print(f"      ↓ child:  {c['child']} ({c['relationship']}) "
                      f"[{cnode['node_type']}]: \"{cnode['claim_text'][:60]}...\"")
            if not parents and not children:
                print(f"      (isolated — no edges)")


# ---------------------------------------------------------------------------
# 5. Edge stability
# ---------------------------------------------------------------------------
def compute_edge_stability(dags, run_claims, all_results):
    """
    For each edge in each run, count how many other runs reproduce it
    (both endpoints matched AND connected by an edge in the other run).

    Returns a DataFrame with one row per edge and an 'n_runs' column (1–N).
    """
    # Build edge sets per run keyed by claim ID
    run_edges = {}
    for name, dag in dags.items():
        run_edges[name] = {(e["parent"], e["child"]) for e in dag["edges"]}

    # Build pairwise claim-ID mappings from the match results
    # pair_mappings[(ra, rb)] = {ra_claim_id: rb_claim_id}
    pair_mappings = {}
    for result in all_results:
        ra, rb = result["run_a"], result["run_b"]
        fwd, bwd = {}, {}
        for a_idx, b_idx in result["matches"]:
            a_id = run_claims[ra][a_idx]["id"]
            b_id = run_claims[rb][b_idx]["id"]
            fwd[a_id] = b_id
            bwd[b_id] = a_id
        pair_mappings[(ra, rb)] = fwd
        pair_mappings[(rb, ra)] = bwd

    # Score each edge — track both "endpoints present" and "edge reproduced"
    records = []
    for name, dag in dags.items():
        for edge in dag["edges"]:
            pid, cid = edge["parent"], edge["child"]
            n_both_present = 0   # other runs where both endpoints matched
            n_reproduced = 0     # other runs where edge actually exists
            for other in sorted(dags):
                if other == name:
                    continue
                mapping = pair_mappings.get((name, other), {})
                mapped_p = mapping.get(pid)
                mapped_c = mapping.get(cid)
                if mapped_p is not None and mapped_c is not None:
                    n_both_present += 1
                    if (mapped_p, mapped_c) in run_edges[other]:
                        n_reproduced += 1
            records.append({
                "run": name,
                "parent": pid,
                "child": cid,
                "relationship": edge["relationship"],
                "n_runs": 1 + n_reproduced,
                "n_runs_both_present": 1 + n_both_present,
            })

    return pd.DataFrame(records)


def print_edge_stability_summary(edge_df, n_runs):
    """Print edge stability distribution to console."""
    total = len(edge_df)
    print(f"\n=== Edge Stability ({total} edges across {n_runs} runs) ===")
    for k in range(1, n_runs + 1):
        c = (edge_df["n_runs"] == k).sum()
        pct = 100 * c / total
        label = {1: "unique to 1 run", n_runs: "found in ALL runs"}.get(
            k, f"in {k} runs"
        )
        print(f"  {label}: {c:3d} ({pct:5.1f}%)")

    # Per-run breakdown
    print("\nPer-run breakdown (edges reproduced in all runs / total):")
    for run_name in sorted(edge_df["run"].unique()):
        run_sub = edge_df[edge_df["run"] == run_name]
        stable = (run_sub["n_runs"] == n_runs).sum()
        print(f"  {run_name}: {stable}/{len(run_sub)} "
              f"({100 * stable / len(run_sub):.0f}%) edges reproduced in all runs")

    # Diagnostic: separate endpoint co-occurrence from wiring
    print("\n--- Diagnostic: Why are edges not reproduced? ---")
    # For edges NOT in all runs, classify the bottleneck
    not_all = edge_df[edge_df["n_runs"] < n_runs]
    if len(not_all) == 0:
        print("  All edges reproduced in all runs — no bottleneck.")
        return

    # For each edge not in all runs, how many other runs had both endpoints?
    # n_runs_both_present - 1 = other runs with both endpoints matched
    # n_runs - 1 = other runs with edge actually present
    # difference = runs where both claims exist but are NOT connected
    n_other = n_runs - 1  # max possible other runs

    # Classify each edge's gap
    endpoint_missing = 0   # edge missing because an endpoint wasn't matched
    wiring_different = 0   # both endpoints present but not connected
    for _, row in not_all.iterrows():
        other_present = row["n_runs_both_present"] - 1  # exclude self
        other_connected = row["n_runs"] - 1              # exclude self
        missing_runs = n_other - other_connected
        # Of those missing runs: how many had both endpoints?
        present_but_not_connected = other_present - other_connected
        endpoint_not_present = missing_runs - present_but_not_connected
        endpoint_missing += endpoint_not_present
        wiring_different += present_but_not_connected

    total_gaps = endpoint_missing + wiring_different
    print(f"  Edges not in all runs: {len(not_all)}")
    print(f"  Total missing edge-run slots: {total_gaps}")
    print(f"    Endpoint not matched in other run:  {endpoint_missing} "
          f"({100 * endpoint_missing / total_gaps:.0f}%)")
    print(f"    Both endpoints present, not wired:  {wiring_different} "
          f"({100 * wiring_different / total_gaps:.0f}%)")

    # Also show: of edges where both endpoints ARE in all runs,
    # what fraction of those edges are in all runs?
    both_in_all = edge_df[edge_df["n_runs_both_present"] == n_runs]
    if len(both_in_all) > 0:
        edges_also_in_all = (both_in_all["n_runs"] == n_runs).sum()
        print(f"\n  Edges where both endpoints are in all {n_runs} runs: {len(both_in_all)}")
        print(f"    Of those, edge also in all runs:   {edges_also_in_all} "
              f"({100 * edges_also_in_all / len(both_in_all):.0f}%)")
        print(f"    Edge missing despite both present:  {len(both_in_all) - edges_also_in_all} "
              f"({100 * (len(both_in_all) - edges_also_in_all) / len(both_in_all):.0f}%)")


# ---------------------------------------------------------------------------
# 6. Metadata consistency
# ---------------------------------------------------------------------------
def compute_metadata_consistency(run_claims, all_results):
    """
    For all matched claim pairs across all run-pairs, check agreement
    on node_type, depth, and claim_type.

    Returns a DataFrame with one row per pair and agreement rates per field.
    """
    fields = ["node_type", "depth", "claim_type"]

    pair_records = []
    for result in all_results:
        ra, rb = result["run_a"], result["run_b"]
        n_matches = len(result["matches"])
        if n_matches == 0:
            continue
        agreements = {f: 0 for f in fields}
        for a_idx, b_idx in result["matches"]:
            ca = run_claims[ra][a_idx]
            cb = run_claims[rb][b_idx]
            for f in fields:
                if ca[f] == cb[f]:
                    agreements[f] += 1
        pair_records.append({
            "run_a": ra,
            "run_b": rb,
            "n_matches": n_matches,
            **{f: 100 * agreements[f] / n_matches for f in fields},
        })

    return pd.DataFrame(pair_records)


def print_metadata_consistency_summary(meta_df):
    """Print metadata consistency to console."""
    fields = ["node_type", "depth", "claim_type"]
    print("\n=== Metadata Consistency (% agreement across matched pairs) ===")
    for f in fields:
        mean_val = meta_df[f].mean()
        min_val = meta_df[f].min()
        max_val = meta_df[f].max()
        print(f"  {f:>12s}: {mean_val:.1f}% mean  (range {min_val:.1f}–{max_val:.1f}%)")


# ---------------------------------------------------------------------------
# 7. Visualizations
# ---------------------------------------------------------------------------
def plot_structural(struct_df, output_dir):
    """Per-run bar charts of structural metrics (2×2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    struct_df.set_index("run")[["roots", "intermediates", "leaves"]].plot(
        kind="bar", stacked=True, ax=ax, color=["#e74c3c", "#f39c12", "#2ecc71"]
    )
    ax.set_title("Nodes by Type", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[0, 1]
    struct_df.set_index("run")[["total_nodes", "total_edges"]].plot(
        kind="bar", ax=ax, color=["#3498db", "#9b59b6"]
    )
    ax.set_title("Total Nodes & Edges", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[1, 0]
    ct_cols = [c for c in struct_df.columns if c.startswith("ct_")]
    ct_labels = [c.replace("ct_", "") for c in ct_cols]
    struct_df.set_index("run")[ct_cols].plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Claim Types per Run", fontweight="bold")
    ax.legend(ct_labels, fontsize=6, loc="upper right")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[1, 1]
    rel_cols = [c for c in struct_df.columns if c.startswith("rel_")]
    rel_labels = [c.replace("rel_", "") for c in rel_cols]
    struct_df.set_index("run")[rel_cols].plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Edge Relationships per Run", fontweight="bold")
    ax.legend(rel_labels, fontsize=7)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(output_dir / "structural_comparison.png", dpi=150)
    plt.close()


def plot_structural_variability(struct_df, output_dir):
    """Grouped bar chart with error bars summarising structural variability."""
    metrics = {
        "Total\nNodes": "total_nodes",
        "Total\nEdges": "total_edges",
        "Roots": "roots",
        "Inter-\nmediates": "intermediates",
        "Leaves": "leaves",
        "Literature\nNodes": "literature_nodes",
        "Max\nDepth": "max_depth",
    }

    labels = list(metrics.keys())
    cols = list(metrics.values())
    means = np.array([struct_df[c].mean() for c in cols])
    mins = np.array([struct_df[c].min() for c in cols])
    maxs = np.array([struct_df[c].max() for c in cols])
    stds = np.array([struct_df[c].std(ddof=1) for c in cols])

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))

    ax.bar(x, means, color="#3498db", edgecolor="black", linewidth=0.5,
           alpha=0.85, zorder=3, width=0.55)
    err_low = means - mins
    err_high = maxs - means
    ax.errorbar(x, means, yerr=[err_low, err_high], fmt="none", ecolor="black",
                capsize=7, capthick=1.5, linewidth=1.5, zorder=4)

    run_colors = plt.cm.Set2(np.linspace(0, 0.8, len(struct_df)))
    for i, (_, row) in enumerate(struct_df.iterrows()):
        vals = [row[c] for c in cols]
        ax.scatter(x, vals, color=run_colors[i], edgecolors="black", linewidth=0.5,
                   s=60, zorder=5, label=row["run"])

    y_pad = max(maxs) * 0.04
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, maxs[i] + y_pad, f"{m:.1f} ± {s:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Structural Variability Across Runs  (mean ± range)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, title="Run", ncol=2)
    ax.set_ylim(0, max(maxs) * 1.22)

    plt.tight_layout()
    fig.savefig(output_dir / "structural_variability.png", dpi=150)
    plt.close()


def plot_claim_stability(claims_df, n_runs, output_dir):
    """
    Stacked bar chart of claim-level stability.
    Left:  dataset-derived claims stacked by node type.
    Right: literature-derived claims stacked by node type.
    """
    dataset_df = claims_df[claims_df["source"] == "dataset"]
    lit_df = claims_df[claims_df["source"] == "literature"]

    node_types = ["root", "intermediate", "leaf"]
    node_colors = {"root": "#e74c3c", "intermediate": "#f39c12", "leaf": "#2ecc71"}
    x = np.arange(1, n_runs + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 2]}
    )

    # ---- Left: dataset-derived claims ----
    bottom = np.zeros(n_runs)
    for nt in node_types:
        subset = dataset_df[dataset_df["node_type"] == nt]
        vals = np.array(
            [(subset["n_runs"] == k).sum() for k in range(1, n_runs + 1)]
        )
        ax1.bar(x, vals, bottom=bottom, label=nt.capitalize(),
                color=node_colors[nt], edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 0:
                ax1.text(x[i], bottom[i] + v / 2, str(v),
                         ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += vals

    totals = np.array(
        [(dataset_df["n_runs"] == k).sum() for k in range(1, n_runs + 1)]
    )
    for i, t in enumerate(totals):
        if t > 0:
            ax1.text(x[i], t + 1, str(t), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="#555555")

    ax1.set_xlabel("Number of Repeats Containing Claim", fontsize=11)
    ax1.set_ylabel("Number of Claims", fontsize=11)
    ax1.set_title("Dataset-Derived Claims", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.legend(title="Node Type", fontsize=9)
    ax1.set_ylim(0, max(totals.max(), 1) * 1.15)

    # ---- Right: literature-derived claims ----
    bottom = np.zeros(n_runs)
    for nt in node_types:
        subset = lit_df[lit_df["node_type"] == nt]
        vals = np.array(
            [(subset["n_runs"] == k).sum() for k in range(1, n_runs + 1)]
        )
        ax2.bar(x, vals, bottom=bottom, label=nt.capitalize(),
                color=node_colors[nt], edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 0:
                ax2.text(x[i], bottom[i] + v / 2, str(v),
                         ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += vals

    totals_lit = np.array(
        [(lit_df["n_runs"] == k).sum() for k in range(1, n_runs + 1)]
    )
    for i, t in enumerate(totals_lit):
        if t > 0:
            ax2.text(x[i], t + 0.3, str(t), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="#555555")

    ax2.set_xlabel("Number of Repeats Containing Claim", fontsize=11)
    ax2.set_ylabel("Number of Claims", fontsize=11)
    ax2.set_title("Literature-Derived Claims", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.legend(title="Node Type", fontsize=9)
    lit_max = totals_lit.max() if len(totals_lit) > 0 and totals_lit.max() > 0 else 1
    ax2.set_ylim(0, lit_max * 1.25)

    fig.suptitle("Claim Stability: How Many Repeats Contain Each Claim?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "claim_stability.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_literature_analysis(dags, claims_df, n_runs, output_dir):
    """Literature-derived claims: count per run, stability, and comparison."""
    lit_counts = {}
    for name, dag in dags.items():
        lit_counts[name] = sum(1 for n in dag["nodes"] if n["source"] == "literature")

    lit_claims = claims_df[claims_df["source"] == "literature"]
    dat_claims = claims_df[claims_df["source"] == "dataset"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Literature claim count per run
    ax = axes[0]
    runs = sorted(lit_counts.keys())
    counts_vals = [lit_counts[r] for r in runs]
    bars = ax.bar(runs, counts_vals, color="#8e44ad", edgecolor="black")
    ax.set_title("Literature-Derived Claims per Run", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts_vals) + 2)
    for bar, val in zip(bars, counts_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontweight="bold")

    # Panel 2: Literature claim stability
    ax = axes[1]
    if len(lit_claims) > 0:
        stab_counts = lit_claims["n_runs"].value_counts().sort_index()
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n_runs))
        for k in range(1, n_runs + 1):
            val = stab_counts.get(k, 0)
            ax.bar(k, val, color=colors[k - 1], edgecolor="black")
            if val > 0:
                ax.text(k, val + 0.1, str(val),
                        ha="center", va="bottom", fontweight="bold")
        ax.set_xticks(range(1, n_runs + 1))
    ax.set_xlabel("Number of Repeats Containing Claim")
    ax.set_ylabel("Literature Claims")
    ax.set_title("Literature Claim Stability", fontweight="bold")

    # Panel 3: Stability comparison — literature vs dataset (%)
    ax = axes[2]
    if len(lit_claims) > 0 or len(dat_claims) > 0:
        xpos = np.arange(1, n_runs + 1)
        width = 0.35
        lit_total = max(len(lit_claims), 1)
        dat_total = max(len(dat_claims), 1)
        lit_pcts = [100 * (lit_claims["n_runs"] == k).sum() / lit_total
                    for k in range(1, n_runs + 1)]
        dat_pcts = [100 * (dat_claims["n_runs"] == k).sum() / dat_total
                    for k in range(1, n_runs + 1)]
        ax.bar(xpos - width / 2, lit_pcts, width, label="Literature",
               color="#8e44ad", edgecolor="black")
        ax.bar(xpos + width / 2, dat_pcts, width, label="Dataset",
               color="#2ecc71", edgecolor="black")
        ax.set_xticks(range(1, n_runs + 1))
        ax.legend()
    ax.set_xlabel("Number of Repeats Containing Claim")
    ax.set_ylabel("% of Claims")
    ax.set_title("Stability: Literature vs Dataset Claims", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "literature_claims_analysis.png", dpi=150)
    plt.close()


def plot_edge_stability(edge_df, n_runs, output_dir):
    """Bar chart of edge stability with diagnostic annotation."""
    x = np.arange(1, n_runs + 1)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_runs))

    vals = [(edge_df["n_runs"] == k).sum() for k in range(1, n_runs + 1)]
    total = len(edge_df)
    stable = vals[-1]
    both_in_all = (edge_df["n_runs_both_present"] == n_runs).sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    str(v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of Repeats Containing Edge", fontsize=11)
    ax.set_ylabel("Number of Edges", fontsize=11)
    ax.set_title(
        f"Edge Stability: {stable}/{total} edges ({100 * stable / total:.0f}%) "
        f"reproduced in all {n_runs} runs",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_ylim(0, max(vals) * 1.2)

    # Diagnostic annotation
    ax.annotate(
        f"{both_in_all} edges have both endpoints in all {n_runs} runs,\n"
        f"but only {stable} ({100 * stable / both_in_all:.0f}%) are wired the same",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#856404"),
    )

    plt.tight_layout()
    fig.savefig(output_dir / "edge_stability.png", dpi=150)
    plt.close()


def compute_wiring_diagnostic(dags, run_claims, all_results):
    """
    For each directed pair (source → target), compute:
    - How many edges from source have both endpoints matched in target
    - Of those, how many are actually connected in target (observed rate)
    - Edge density of target (chance rate)
    """
    run_edges = {}
    for name, dag in dags.items():
        run_edges[name] = {(e["parent"], e["child"]) for e in dag["edges"]}

    pair_mappings = {}
    for result in all_results:
        ra, rb = result["run_a"], result["run_b"]
        fwd, bwd = {}, {}
        for a_idx, b_idx in result["matches"]:
            a_id = run_claims[ra][a_idx]["id"]
            b_id = run_claims[rb][b_idx]["id"]
            fwd[a_id] = b_id
            bwd[b_id] = a_id
        pair_mappings[(ra, rb)] = fwd
        pair_mappings[(rb, ra)] = bwd

    records = []
    runs = sorted(dags)
    for source in runs:
        for target in runs:
            if source == target:
                continue
            mapping = pair_mappings.get((source, target), {})
            n_testable = 0
            n_reproduced = 0
            for edge in dags[source]["edges"]:
                mapped_p = mapping.get(edge["parent"])
                mapped_c = mapping.get(edge["child"])
                if mapped_p is not None and mapped_c is not None:
                    n_testable += 1
                    if (mapped_p, mapped_c) in run_edges[target]:
                        n_reproduced += 1

            n_t = len(dags[target]["nodes"])
            n_e = len(dags[target]["edges"])
            chance = n_e / (n_t * (n_t - 1)) if n_t > 1 else 0

            records.append({
                "source": source,
                "target": target,
                "n_testable": n_testable,
                "n_reproduced": n_reproduced,
                "observed_rate": n_reproduced / n_testable if n_testable > 0 else 0,
                "chance_rate": chance,
            })

    return pd.DataFrame(records)


def plot_wiring_diagnostic(wiring_df, output_dir):
    """
    Compare observed wiring reproduction rate to chance (edge density).
    Shows that even though exact edge reproduction is low, it's far above random.
    """
    observed = wiring_df["observed_rate"].values * 100
    chance = wiring_df["chance_rate"].values * 100

    mean_obs = observed.mean()
    mean_chance = chance.mean()
    fold = mean_obs / mean_chance if mean_chance > 0 else float("inf")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array([0, 1])
    width = 0.45

    # Bars for mean rates
    bars = ax.bar(x, [mean_chance, mean_obs],
                  color=["#e0e0e0", "#2ecc71"], edgecolor="black",
                  linewidth=0.5, width=width, zorder=3)

    # Overlay individual pair values as dots
    pair_colors = plt.cm.Set2(np.linspace(0, 0.8, len(wiring_df)))
    for i, (_, row) in enumerate(wiring_df.iterrows()):
        label = f"{row['source']}→{row['target']}"
        ax.scatter(1, row["observed_rate"] * 100, color=pair_colors[i % len(pair_colors)],
                   edgecolors="black", linewidth=0.4, s=40, zorder=5)

    # Annotate
    ax.text(0, mean_chance + 0.5, f"{mean_chance:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#666")
    ax.text(1, mean_obs + 0.5, f"{mean_obs:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Random chance\n(edge density)", "Observed\nreproduction rate"],
                       fontsize=11)
    ax.set_ylabel("% of testable edges reproduced", fontsize=11)
    ax.set_title(
        f"Wiring Reproduction vs Chance: {fold:.0f}× above random\n"
        f"(edges where both endpoints are matched in the other run)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, max(observed.max(), mean_obs) * 1.2)

    plt.tight_layout()
    fig.savefig(output_dir / "wiring_diagnostic.png", dpi=150)
    plt.close()


def plot_metadata_consistency(meta_df, output_dir):
    """Bar chart of metadata agreement rates with per-pair dots."""
    fields = ["node_type", "depth", "claim_type"]
    labels = ["Node Type\n(root / intermediate / leaf)",
              "Depth",
              "Claim Type\n(empirical / statistical / ...)"]

    means = [meta_df[f].mean() for f in fields]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fields))
    bars = ax.bar(x, means, color="#3498db", edgecolor="black", linewidth=0.5,
                  alpha=0.85, width=0.5, zorder=3)

    # Overlay individual pair values as dots
    pair_colors = plt.cm.Set2(np.linspace(0, 0.8, len(meta_df)))
    for i, (_, row) in enumerate(meta_df.iterrows()):
        pair_label = f"{row['run_a']}–{row['run_b']}"
        vals = [row[f] for f in fields]
        ax.scatter(x, vals, color=pair_colors[i], edgecolors="black",
                   linewidth=0.5, s=55, zorder=5,
                   label=pair_label if i < 10 else None)

    # Annotate mean values
    for i, m in enumerate(means):
        ax.text(i, m + 1.2, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% Agreement", fontsize=11)
    ax.set_title("Metadata Consistency Across Matched Claims",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=7, title="Run pair", ncol=2)

    plt.tight_layout()
    fig.savefig(output_dir / "metadata_consistency.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare variance across repeated DAG extractions."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--claude", action="store_true",
        help="Use Claude Opus 4.6 for pairwise claim matching (requires ANTHROPIC_API_KEY)",
    )
    mode_group.add_argument(
        "--embed", action="store_true",
        help="Use sentence-transformers cosine similarity for matching",
    )
    args = parser.parse_args()

    use_claude = args.claude

    print("=" * 70)
    print("DAG Variance Analysis — Repeated Claim Extraction")
    mode_label = (
        f"Claude {CLAUDE_MODEL} (pairwise)" if use_claude
        else f"sentence-transformers ({MODEL_NAME})"
    )
    print(f"Matching mode: {mode_label}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load
    dags = load_dags()
    n_runs = len(dags)
    print(f"\nLoaded {n_runs} runs: {', '.join(RUN_NAMES)}")

    # 2. Structural summary
    struct_df = structural_summary(dags)
    print("\n=== Structural Summary ===")
    print(struct_df[["run", "total_nodes", "total_edges", "roots", "intermediates",
                      "leaves", "literature_nodes", "max_depth", "mean_depth",
                      "n_excluded", "glossary_terms"]].to_string(index=False))

    # 3. Pairwise claim matching
    if use_claude:
        run_claims, all_results = run_pairwise_claude(dags)
    else:
        run_claims, all_results = run_pairwise_embed(dags)

    # 4. Claim-level stability
    claims_df = compute_claim_stability(run_claims, all_results)
    print_claim_stability_summary(claims_df, n_runs)

    # 5. Edge stability
    edge_df = compute_edge_stability(dags, run_claims, all_results)
    print_edge_stability_summary(edge_df, n_runs)

    # 6. Wiring diagnostic
    wiring_df = compute_wiring_diagnostic(dags, run_claims, all_results)
    mean_obs = wiring_df["observed_rate"].mean() * 100
    mean_chance = wiring_df["chance_rate"].mean() * 100
    fold = mean_obs / mean_chance if mean_chance > 0 else float("inf")
    print(f"\n=== Wiring Diagnostic ===")
    print(f"  Mean observed reproduction rate: {mean_obs:.1f}%")
    print(f"  Mean chance rate (edge density): {mean_chance:.1f}%")
    print(f"  Fold above chance:               {fold:.0f}×")

    # 7. Metadata consistency
    meta_df = compute_metadata_consistency(run_claims, all_results)
    print_metadata_consistency_summary(meta_df)

    # 8. Literature details (console only)
    print_literature_details(dags, n_runs)

    # 9. Visualizations
    print("\nGenerating visualizations ...")
    plot_structural(struct_df, OUTPUT_DIR)
    plot_structural_variability(struct_df, OUTPUT_DIR)
    plot_claim_stability(claims_df, n_runs, OUTPUT_DIR)
    plot_edge_stability(edge_df, n_runs, OUTPUT_DIR)
    plot_wiring_diagnostic(wiring_df, OUTPUT_DIR)
    plot_metadata_consistency(meta_df, OUTPUT_DIR)
    plot_literature_analysis(dags, claims_df, n_runs, OUTPUT_DIR)

    # 9. Final summary
    total_claims = len(claims_df)
    stable_claims = (claims_df["n_runs"] == n_runs).sum()
    volatile_claims = (claims_df["n_runs"] == 1).sum()
    total_edges = len(edge_df)
    stable_edges = (edge_df["n_runs"] == n_runs).sum()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Matching mode:           {mode_label}")
    print(f"  Runs compared:           {n_runs}")
    print(f"  Claims:  {stable_claims}/{total_claims} "
          f"({100 * stable_claims / total_claims:.0f}%) reproduced in all runs")
    print(f"  Edges:   {stable_edges}/{total_edges} "
          f"({100 * stable_edges / total_edges:.0f}%) reproduced in all runs")
    fields = ["node_type", "depth", "claim_type"]
    for f in fields:
        print(f"  {f} agreement:  {meta_df[f].mean():.1f}%")
    print(f"  Wiring: {fold:.0f}× above chance")
    print(f"\nPlots saved to: {OUTPUT_DIR}/")
    print("  structural_comparison.png      — per-run breakdown")
    print("  structural_variability.png     — mean ± range across runs")
    print("  claim_stability.png            — claims by repeat count & node type")
    print("  edge_stability.png             — edges by repeat count")
    print("  wiring_diagnostic.png          — observed vs chance wiring rate")
    print("  metadata_consistency.png       — node_type / depth / claim_type agreement")
    print("  literature_claims_analysis.png — literature claim stability")
    if use_claude:
        print("  claude_pairwise_results.json   — per-pair match details (audit trail)")


if __name__ == "__main__":
    main()
