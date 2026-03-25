# Hierarchical Claim Extraction Pipeline

This project extracts verifiable scientific claims from biology research papers and organizes them into a hierarchical directed acyclic graph (DAG) for reproducibility evaluation by AI research agents (Finch, Kosmos).

## Pipeline Overview

The pipeline has five stages, run sequentially. Each stage has a dedicated prompt file in this directory.

| Stage | Prompt File | Input | Output | Run Per |
|-------|------------|-------|--------|---------|
| **0** | `stage0_dataset_link_extraction_prompt.txt` | Paper PDF | Dataset links CSV | Paper |
| **1** | `stage1_dataset_identification_prompt.txt` | Paper PDF + ALL dataset repo links | Combined `dataset_profile.md` + `section_inventory.csv` | Paper |
| **2** | `stage2_claim_extraction_prompt.txt` | Paper PDF + combined profile + inventory | `claims.csv` + `key_terms.csv` | Paper |
| **3** | `stage3_claim_dag_construction_prompt.txt` | Paper PDF + dataset profile + claims CSV | `claim_dag.json` | Paper |
| **4** | `stage4_analysis_protocol_prompt.txt` | Paper PDF + `claim_dag.json` + code repo | Updated `claim_dag.json` | Paper |

**Important:** Use `stage2_claim_extraction_prompt.txt` (v1), NOT `stage2_claim_extraction_prompt_v2.txt`.

## Workflow

Given a paper PDF:

1. **Stage 0** — Run once per paper. Read the paper and extract all dataset repository links and accession IDs. Output: a CSV of dataset links with accession IDs, types, and roles (primary vs reanalyzed).

2. **Stage 1** — Run once per paper. Provide ALL dataset links from Stage 0 at once. Output: a single combined `dataset_profile.md` covering all datasets and a single `section_inventory.csv` with a `dataset_source` column indicating which dataset(s) each section derives from.

3. **Stage 2** — Run once per paper. Provide the paper PDF, the combined dataset profile, and the section inventory. Output: `claims.csv` (flat atomic claims that may draw on any of the paper's datasets) and `key_terms.csv`.

4. **Stage 3** — Run once per paper. Provide the paper PDF, the dataset profile, and the claims CSV. Output: `claim_dag.json` (hierarchical DAG with root, intermediate, and leaf claims).

5. **Stage 4** — Run once per paper. Extracts code repository URLs from the paper text, downloads the repo, reads analysis scripts, and enhances `analysis_detail` fields in `claim_dag.json` with code-informed executable protocols. **Auto-skipped** if no code repository is found in the paper. The original DAG is backed up as `claim_dag_pre_stage4.json`.

## Directory Structure

Each paper gets a subdirectory under `Papers_claims/` named with a short snake_case identifier. All outputs go in the same flat directory regardless of how many datasets the paper has:

```
Papers_claims/
  paper_short_name/
    paper.pdf                    # The source paper
    dataset_links.csv            # Stage 0 output
    dataset_profile.md           # Stage 1 output (combined profile for all datasets)
    section_inventory.csv        # Stage 1 output (with dataset_source column)
    claims.csv                   # Stage 2 output
    key_terms.csv                # Stage 2 output
    claim_dag.json               # Stage 3 output (updated in-place by Stage 4)
    claim_dag_pre_stage4.json    # Stage 3 original (backup before Stage 4)
```

## Key Design Decisions

- **DAG, not tree:** Claims can have multiple parents (e.g., a diversity finding supporting both brain microglia and kidney macrophage conclusions).
- **Nodes are claims** (positive assertions), not research questions.
- **Leaf claims** are atomic, falsifiable, dataset-sufficient, and method-agnostic. They are preserved exactly from Stage 2.
- **Edge types:** supports, quantifies, qualifies, contrasts, interprets, enables.
- **Depth** = shortest path to any root node. Replaces importance_tier from Stage 2.
- **Dataset sufficiency** is enforced at leaf level only. Intermediate and root nodes can be broader.
- **Literature-derived claims** are included when a prior finding serves as a premise, link, or frame that at least one DAG node depends on (foundational premises, comparative anchors, mechanistic chain links, interpretation frames, contrasts). Flagged with `source: "literature"`. Stage 3 uses a two-phase process: Phase A enumerates all candidates (with a cross-check against intermediate nodes), then Phase B filters. The corroborative exclusion distinguishes observations from interpretations — if the data shows a change but the literature explains its significance, the literature claim is structurally necessary.
- **Dataset accessions** (e.g., GEO IDs) are included at every node.
- **supporting_quote** is a strictly verbatim excerpt from the paper for every node, used for human auditing.
- **experiment_type** and **analysis_type** are concise categorical labels at each node.
- **experiment_detail** and **analysis_detail** are expanded narrative fields describing what the authors did. `analysis_detail` is written at sufficient detail to serve as a prompt for an LLM agent to reproduce the analysis. Present on leaf nodes (always) and intermediate nodes (only when a distinct experiment/analysis produced them, not when they merely synthesize children). Null on root and literature-derived nodes.
- **Paper-level glossary** in the JSON replaces per-claim key_terms.
- **Stage 4 code enhancement** fetches the paper's GitHub/GitLab/Bitbucket code repository and uses it to rewrite `analysis_detail` into executable-grade protocols. Adds `analysis_detail_enhanced` (boolean), `code_scripts` (list of file:line-range strings) per node, and a top-level `code_repository` metadata object. Immutable fields are validated post-enhancement and restored if Claude modifies them.

## Output Format Reference

### Stage 0: `dataset_links.csv`
```
dataset_id,dataset_type,repository,accession_or_url,role,description
```

### Stage 2: `claims.csv`
```
claim_id,claim_text,claim_type,importance_tier,evidence_location,dataset_link,supporting_quote,key_terms
```

### Stage 3: `claim_dag.json`
```json
{
  "paper": {"title": "", "authors": [], "venue": "", "year": ""},
  "glossary": {"TERM": "definition"},
  "nodes": [{"id": "R1", "claim_text": "", "claim_type": "", "node_type": "root|intermediate|leaf", "depth": 0, "source": "dataset|literature", "datasets": [], "dataset_accessions": [], "experiment_type": "", "experiment_detail": "", "analysis_type": "", "analysis_detail": "", "evidence_location": "", "supporting_quote": "", "dataset_link": ""}],
  "edges": [{"child": "C2", "parent": "C1", "relationship": "supports|quantifies|qualifies|contrasts|interprets|enables"}],
  "literature_candidates": [{"candidate_id": 1, "claim_text": "", "scenario": "", "source_sentence": "", "connected_nodes": [], "disposition": "included as LL1|excluded", "reason": ""}],
  "excluded_borderline_claims": [{"claim_text": "", "reason": ""}]
}
```

### Stage 4 additions to `claim_dag.json`

Top-level metadata added:
```json
{
  "code_repository": {
    "url": "https://github.com/owner/repo",
    "branch": "main",
    "fetched_date": "2026-03-24",
    "files_read": ["analysis/01_preprocess.R", "analysis/02_differential.R"]
  }
}
```

Per-node fields added (only on nodes with non-null `analysis_detail`):
```json
{
  "analysis_detail_enhanced": true,
  "code_scripts": ["analysis/02_differential.R:45-120"]
}
```

Node ID conventions: `R1, R2...` (roots), `I1, I2...` (intermediates), `L1, L2...` (dataset-derived leaves), `LL1, LL2...` (literature-derived leaves).

## Downstream Analysis

After Stage 3 (or Stage 4 if a code repo is available), claim DAGs are used by:
- **Finch_analysis/** — Jupyter notebooks submitting claims to the Edison Scientific API for verification
- **Kosmos_analysis/** — Kosmos world model verification runs
- **Combined_analysis/** — Python scripts for aggregation, visualization, and grading (`build_*.py`)
