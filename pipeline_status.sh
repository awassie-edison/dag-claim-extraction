#!/bin/bash
#
# Check progress of running pipeline agents.
# Run this in a separate terminal while run_pipeline.sh is executing.
#
# Usage:
#   ./pipeline_status.sh              # one-time check
#   watch ./pipeline_status.sh        # auto-refresh every 2 seconds
#
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/pipeline_logs/latest"

if [ ! -d "$LOG_DIR" ]; then
    echo "No pipeline run found. Start one with ./run_pipeline.sh"
    exit 1
fi

echo "=== Pipeline Status ($(date '+%H:%M:%S')) ==="
echo "Log directory: $(readlink -f "$LOG_DIR" 2>/dev/null || echo "$LOG_DIR")"
echo ""

for paper_log_dir in "$LOG_DIR"/*/; do
    [ -d "$paper_log_dir" ] || continue
    paper_name="$(basename "$paper_log_dir")"

    paper_dir="${PROJECT_ROOT}/Papers_claims/${paper_name}"

    # Determine pipeline completion from status.log
    pipeline_done="no"
    if grep -q "PIPELINE COMPLETE: ${paper_name}" "$LOG_DIR/status.log" 2>/dev/null; then
        pipeline_done="yes"
    fi

    # Determine current stage from output files
    if [ "$pipeline_done" = "yes" ] && [ -f "$paper_dir/claim_dag.json" ]; then
        stage="COMPLETE"
    elif [ -f "$paper_dir/claim_dag.json" ]; then
        stage="Stage 3 (validating)"
    elif [ -f "$paper_dir/claims.csv" ]; then
        stage="Stage 3 (building DAG)"
    elif [ -f "$paper_dir/section_inventory.csv" ]; then
        stage="Stage 2 (extracting claims)"
    elif [ -f "$paper_dir/dataset_profile.md" ]; then
        stage="Stage 2 (starting)"
    elif [ -f "$paper_dir/dataset_links.csv" ]; then
        stage="Stage 1 (identifying sections)"
    else
        stage="Stage 0 (extracting dataset links)"
    fi

    # Determine which stage log is most recent / active
    latest_stage_log=""
    for s in 3 2 1 0; do
        if [ -f "$paper_log_dir/stage${s}.log" ]; then
            latest_stage_log="stage${s}.log"
            break
        fi
    done

    # Per-stage completion markers from status.log
    stages_done=""
    for s in 0 1 2 3; do
        if grep -q "STAGE ${s} COMPLETE: ${paper_name}" "$LOG_DIR/status.log" 2>/dev/null; then
            stages_done="${stages_done}${s}"
        fi
    done

    if [ "$pipeline_done" = "yes" ]; then
        printf "  %-20s  [DONE]       %s  (stages: 0-3 all complete)\n" "$paper_name" "$stage"
    else
        log_size=""
        if [ -n "$latest_stage_log" ]; then
            log_size=$(du -h "$paper_log_dir/$latest_stage_log" 2>/dev/null | cut -f1)
        fi
        stages_info="done: ${stages_done:-none}"
        printf "  %-20s  [%-6s]     %s  (%s)\n" "$paper_name" "$log_size" "$stage" "$stages_info"
    fi
done

echo ""

# Summary
total=$(find "$LOG_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
done_count=$(grep -c "PIPELINE COMPLETE" "$LOG_DIR/status.log" 2>/dev/null || echo "0")
echo "Progress: $done_count / $total papers complete"
