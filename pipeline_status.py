#!/usr/bin/env python3
"""
Display the current status of the pipeline.

Usage:
    python pipeline_status.py          # show status once
    python pipeline_status.py --watch  # refresh every 5 seconds
"""

import argparse
import json
import sys
import time
from pathlib import Path

STATUS_FILE = Path(__file__).resolve().parent / "pipeline_status.json"

STAGE_NAMES = {
    "0": "Dataset links",
    "1": "Dataset ID",
    "2": "Claims",
    "3": "DAG",
    "4": "Code enhance",
}


def show_status():
    if not STATUS_FILE.exists():
        print("No pipeline_status.json found — pipeline has not been started.")
        return False

    with open(STATUS_FILE) as f:
        status = json.load(f)

    print("\033[2J\033[H", end="")  # clear screen
    print("=" * 70)
    print(f"Pipeline Status  |  Model: {status.get('model', '?')}  |  "
          f"Temp: {status.get('temperature', '?')}")
    print(f"Started: {status.get('pipeline_start', '?')}")
    if "pipeline_end" in status:
        print(f"Finished: {status['pipeline_end']}")
    print("=" * 70)

    papers = status.get("papers", {})
    if not papers:
        print("  No papers registered.")
        return True

    # Header
    print(f"\n  {'Paper':<15s}  {'Status':<10s}  ", end="")
    for sn in ["0", "1", "2", "3", "4"]:
        print(f"  {STAGE_NAMES[sn]:<14s}", end="")
    print(f"  {'Total':>8s}")
    print("  " + "-" * 106)

    all_done = True
    for name in sorted(papers):
        paper = papers[name]
        p_status = paper["status"]
        if p_status != "done":
            all_done = False

        # Status emoji/symbol
        status_str = {
            "pending": "⏳ pending",
            "running": "🔄 running",
            "done": "✅ done",
            "failed": "❌ failed",
        }.get(p_status, p_status)

        print(f"  {name:<15s}  {status_str:<10s}  ", end="")

        stages = paper.get("stages", {})
        for sn in ["0", "1", "2", "3", "4"]:
            if sn in stages:
                s = stages[sn]
                s_status = s["status"]
                if s_status == "done":
                    elapsed = s.get("elapsed_s", 0)
                    print(f"  ✅ {elapsed:>5.0f}s      ", end="")
                elif s_status == "running":
                    print(f"  🔄 running     ", end="")
                elif s_status == "failed":
                    print(f"  ❌ failed      ", end="")
                else:
                    print(f"  {'?':<14s}", end="")
            else:
                print(f"  {'—':<14s}", end="")

        total = paper.get("total_elapsed_s")
        if total is not None:
            print(f"  {total:>7.0f}s")
        else:
            print(f"  {'—':>8s}")

    # Token summary
    total_in = 0
    total_out = 0
    for paper in papers.values():
        for stage in paper.get("stages", {}).values():
            total_in += stage.get("tokens_in", 0)
            total_out += stage.get("tokens_out", 0)

    n_done = sum(1 for p in papers.values() if p["status"] == "done")
    n_total = len(papers)

    print(f"\n  Progress: {n_done}/{n_total} papers complete")
    print(f"  Tokens:   {total_in:,} input / {total_out:,} output")

    return all_done


def main():
    parser = argparse.ArgumentParser(description="Display pipeline status.")
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every 5 seconds until pipeline completes")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                done = show_status()
                if done:
                    print("\n  Pipeline complete.")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Stopped watching.")
    else:
        show_status()


if __name__ == "__main__":
    main()
