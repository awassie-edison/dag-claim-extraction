#!/bin/bash
#
# Run the claim extraction pipeline (Stage 0 -> 1 -> 2 -> 3) on one or more papers.
# Each paper is processed in parallel. Each stage runs as a separate agent with a
# fresh context window, reading prior stage outputs from disk.
#
# Usage:
#   ./run_pipeline.sh paper_dir1 [paper_dir2 ...]
#
# Each paper_dir should be a subdirectory under Papers_claims/ containing:
#   - paper.pdf (or any .pdf file)
#
# Example:
#   ./run_pipeline.sh Papers_claims/CAG_Repeat_HD Papers_claims/tabula_muris_senis
#
# Or process all papers:
#   ./run_pipeline.sh Papers_claims/*/
#
set -euo pipefail

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
MAX_PARALLEL=5           # Max concurrent paper-level pipelines
MAX_TURNS_PER_STAGE=50   # Max agentic turns per stage agent
LOG_DIR="${PROJECT_ROOT}/pipeline_logs/$(date +%Y%m%d_%H%M%S)"
ALLOWED_TOOLS="Read,Write,Edit,Glob,Grep,Bash(python3 *),Bash(ls *),Bash(mkdir *),WebFetch,WebSearch"

mkdir -p "$LOG_DIR"

echo "=== Claim Extraction Pipeline (per-stage agents) ==="
echo "Project root: $PROJECT_ROOT"
echo "Log directory: $LOG_DIR"
echo "Max parallel papers: $MAX_PARALLEL"
echo "Max turns per stage: $MAX_TURNS_PER_STAGE"
echo ""

if [ $# -eq 0 ]; then
    echo "Usage: $0 paper_dir1 [paper_dir2 ...]"
    echo ""
    echo "Available paper directories:"
    ls -d "$PROJECT_ROOT"/Papers_claims/*/ 2>/dev/null | while read -r d; do
        basename "$d"
    done
    exit 1
fi

# --- Function: run a single stage for a paper ---
run_stage() {
    local stage_num="$1"
    local paper_name="$2"
    local paper_path="$3"
    local pdf_file="$4"
    local prompt="$5"
    local log_file="$6"

    echo "  [${paper_name}] Starting Stage ${stage_num}..."

    claude -p "$prompt" \
        --model "opus[1m]" \
        --allowedTools "$ALLOWED_TOOLS" \
        --max-turns "$MAX_TURNS_PER_STAGE" \
        > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  [${paper_name}] Stage ${stage_num} FAILED (exit code ${exit_code})" | tee -a "$LOG_DIR/status.log"
        return 1
    fi
    echo "  [${paper_name}] Stage ${stage_num} complete."
    return 0
}

# --- Function: run all stages sequentially for a single paper ---
run_paper_pipeline() {
    local paper_path="$1"
    local paper_name="$(basename "$paper_path")"
    local pdf_file
    pdf_file=$(find "$paper_path" -maxdepth 1 -name "*.pdf" -type f | head -1)

    local paper_log_dir="${LOG_DIR}/${paper_name}"
    mkdir -p "$paper_log_dir"

    echo "[${paper_name}] Pipeline starting (PDF: $(basename "$pdf_file"))"

    # --- Stage 0: Dataset Link Extraction ---
    run_stage 0 "$paper_name" "$paper_path" "$pdf_file" \
"Read the prompt from ${PROJECT_ROOT}/stage0_dataset_link_extraction_prompt.txt and follow its instructions exactly.

Your task: Extract all dataset repository links and accession IDs from the paper.

Input:
- Paper PDF: ${pdf_file}

Output:
- Save as: ${paper_path}/dataset_links.csv

Read the CLAUDE.md at ${PROJECT_ROOT}/CLAUDE.md for context on the pipeline and output format conventions." \
        "$paper_log_dir/stage0.log" || return 1

    # Verify Stage 0 output
    if [ ! -f "${paper_path}/dataset_links.csv" ]; then
        echo "  [${paper_name}] Stage 0 did not produce dataset_links.csv" | tee -a "$LOG_DIR/status.log"
        return 1
    fi
    echo "STAGE 0 COMPLETE: ${paper_name} ($(date))" >> "$LOG_DIR/status.log"

    # --- Stage 1: Dataset Identification ---
    run_stage 1 "$paper_name" "$paper_path" "$pdf_file" \
"Read the prompt from ${PROJECT_ROOT}/stage1_dataset_identification_prompt.txt and follow its instructions exactly.

Your task: Create a combined dataset profile and section inventory for all datasets in this paper.

Inputs:
- Paper PDF: ${pdf_file}
- Dataset links from Stage 0: ${paper_path}/dataset_links.csv

Outputs:
- Save combined dataset profile as: ${paper_path}/dataset_profile.md
- Save section inventory as: ${paper_path}/section_inventory.csv (must include a dataset_source column)

Provide ALL dataset links from dataset_links.csv at once. Produce a single combined profile covering all datasets.

Read the CLAUDE.md at ${PROJECT_ROOT}/CLAUDE.md for context on the pipeline and output format conventions." \
        "$paper_log_dir/stage1.log" || return 1

    # Verify Stage 1 outputs
    if [ ! -f "${paper_path}/dataset_profile.md" ] || [ ! -f "${paper_path}/section_inventory.csv" ]; then
        echo "  [${paper_name}] Stage 1 did not produce expected outputs" | tee -a "$LOG_DIR/status.log"
        return 1
    fi
    echo "STAGE 1 COMPLETE: ${paper_name} ($(date))" >> "$LOG_DIR/status.log"

    # --- Stage 2: Claim Extraction ---
    run_stage 2 "$paper_name" "$paper_path" "$pdf_file" \
"Read the prompt from ${PROJECT_ROOT}/stage2_claim_extraction_prompt.txt (NOT any v2 version) and follow its instructions exactly.

Your task: Extract all distinct, verifiable claims from the paper that are grounded in the authors' analysis of the paper's datasets.

Inputs:
- Paper PDF: ${pdf_file}
- Combined dataset profile: ${paper_path}/dataset_profile.md
- Section inventory: ${paper_path}/section_inventory.csv

Outputs:
- Save claims as: ${paper_path}/claims.csv
- Save key terms as: ${paper_path}/key_terms.csv

Run once for the entire paper (all datasets together).

Read the CLAUDE.md at ${PROJECT_ROOT}/CLAUDE.md for context on the pipeline and output format conventions." \
        "$paper_log_dir/stage2.log" || return 1

    # Verify Stage 2 outputs
    if [ ! -f "${paper_path}/claims.csv" ]; then
        echo "  [${paper_name}] Stage 2 did not produce claims.csv" | tee -a "$LOG_DIR/status.log"
        return 1
    fi
    echo "STAGE 2 COMPLETE: ${paper_name} ($(date))" >> "$LOG_DIR/status.log"

    # --- Stage 3: DAG Construction ---
    run_stage 3 "$paper_name" "$paper_path" "$pdf_file" \
"Read the prompt from ${PROJECT_ROOT}/stage3_claim_dag_construction_prompt.txt and follow its instructions exactly.

Your task: Organize the extracted claims into a hierarchical directed acyclic graph (DAG).

Inputs:
- Paper PDF: ${pdf_file}
- Combined dataset profile: ${paper_path}/dataset_profile.md
- Claims from Stage 2: ${paper_path}/claims.csv

Output:
- Save as: ${paper_path}/claim_dag.json

After constructing the DAG, validate it:
- Check all leaf claims from claims.csv are present
- Check no cycles exist
- Check no orphan nodes
- Check all root nodes have zero parent edges
- Check all depths are computed correctly

Print a summary: number of root/intermediate/leaf/literature-derived nodes, max depth, number of edges, number of datasets.

Read the CLAUDE.md at ${PROJECT_ROOT}/CLAUDE.md for context on the pipeline and output format conventions." \
        "$paper_log_dir/stage3.log" || return 1

    # Verify Stage 3 output
    if [ ! -f "${paper_path}/claim_dag.json" ]; then
        echo "  [${paper_name}] Stage 3 did not produce claim_dag.json" | tee -a "$LOG_DIR/status.log"
        return 1
    fi
    echo "STAGE 3 COMPLETE: ${paper_name} ($(date))" >> "$LOG_DIR/status.log"

    echo "PIPELINE COMPLETE: ${paper_name} ($(date))" >> "$LOG_DIR/status.log"
    echo "[${paper_name}] Pipeline complete."
}

# --- Process each paper ---
for paper_path in "$@"; do
    # Resolve to absolute path
    if [[ "$paper_path" != /* ]]; then
        paper_path="${PROJECT_ROOT}/${paper_path}"
    fi
    # Remove trailing slash
    paper_path="${paper_path%/}"
    paper_name="$(basename "$paper_path")"

    # Verify directory exists and has a PDF
    if [ ! -d "$paper_path" ]; then
        echo "WARNING: Directory not found: $paper_path — skipping"
        continue
    fi

    pdf_file=$(find "$paper_path" -maxdepth 1 -name "*.pdf" -type f | head -1)
    if [ -z "$pdf_file" ]; then
        echo "WARNING: No PDF found in $paper_path — skipping"
        continue
    fi

    # Launch paper pipeline in background
    run_paper_pipeline "$paper_path" &

    # Limit concurrency
    while (( $(jobs -r -p | wc -l) >= MAX_PARALLEL )); do
        wait -n 2>/dev/null || true
    done
done

# --- Create a convenience symlink for latest run ---
ln -sfn "$LOG_DIR" "${PROJECT_ROOT}/pipeline_logs/latest"

echo ""
echo "All paper pipelines launched. Waiting for completion..."
echo ""
echo "Monitor progress in another terminal with:"
echo "  tail -f ${PROJECT_ROOT}/pipeline_logs/latest/*/*.log"
echo "  cat ${PROJECT_ROOT}/pipeline_logs/latest/status.log"
echo "  # or check stage progress:"
echo "  ${PROJECT_ROOT}/pipeline_status.sh"
echo ""

wait

echo ""
echo "=== Pipeline Complete ==="
echo "Results:"
for paper_path in "$@"; do
    paper_path="${paper_path%/}"
    paper_name="$(basename "$paper_path")"
    if [[ "$paper_path" != /* ]]; then
        paper_path="${PROJECT_ROOT}/${paper_path}"
    fi

    dag_file="${paper_path}/claim_dag.json"
    if [ -f "$dag_file" ]; then
        node_count=$(python3 -c "import json; d=json.load(open('$dag_file')); print(len(d['nodes']))" 2>/dev/null || echo "?")
        echo "  $paper_name: claim_dag.json ($node_count nodes)"
    else
        echo "  $paper_name: NO claim_dag.json (check $LOG_DIR/${paper_name}/)"
    fi
done
echo ""
echo "Logs: $LOG_DIR/"
