#!/home/ubuntu/sprite-venv/bin/python3

"""Compute window/span combinations from config.

Reads the experiment config to determine which time windows are enabled
(e.g. 1mo, 3mo, 6mo) and computes the list of (window, span) pairs based
on the scan date range. Outputs a JSON marker with the full plan.

Corresponds to orchestrator Step B: prepare_spans().
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def months_in_range(start_str, end_str):
    """Yield YYYY-MM strings for each month touched by [start, end)."""
    start = datetime.strptime(str(start_str), "%Y-%m-%d")
    end = datetime.strptime(str(end_str), "%Y-%m-%d")
    current = start.replace(day=1)
    while current < end:
        yield current.strftime("%Y-%m")
        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def quarters_in_range(start_str, end_str):
    """Yield YYYY-QN strings for each quarter touched by [start, end)."""
    start = datetime.strptime(str(start_str), "%Y-%m-%d")
    end = datetime.strptime(str(end_str), "%Y-%m-%d")
    current = start.replace(day=1)
    while current < end:
        q = (current.month - 1) // 3 + 1
        yield f"{current.year}-Q{q}"
        # Advance to next quarter
        new_month = ((current.month - 1) // 3 + 1) * 3 + 1
        if new_month > 12:
            current = current.replace(year=current.year + 1, month=new_month - 12)
        else:
            current = current.replace(month=new_month)


def years_in_range(start_str, end_str):
    """Yield YYYY strings for each year touched by [start, end)."""
    start = datetime.strptime(str(start_str), "%Y-%m-%d")
    end = datetime.strptime(str(end_str), "%Y-%m-%d")
    for y in range(start.year, end.year + 1):
        yield str(y)


def compute_spans(window_name, start_str, end_str):
    """Compute span labels for a given window type and date range."""
    spans = []
    if window_name == "1mo":
        for m in months_in_range(start_str, end_str):
            spans.append({
                "window": window_name,
                "span": f"span_{m}",
                "start": str(start_str),
                "end": str(end_str),
            })
    elif window_name in ("3mo", "6mo", "12mo", "24mo", "48mo"):
        # Multi-month windows: still enumerate by month
        for m in months_in_range(start_str, end_str):
            spans.append({
                "window": window_name,
                "span": f"span_{m}",
                "start": str(start_str),
                "end": str(end_str),
            })
    elif window_name == "1q":
        for q in quarters_in_range(start_str, end_str):
            spans.append({
                "window": window_name,
                "span": f"span_{q}",
                "start": str(start_str),
                "end": str(end_str),
            })
    elif window_name == "1y":
        for y in years_in_range(start_str, end_str):
            spans.append({
                "window": window_name,
                "span": f"span_{y}",
                "start": str(start_str),
                "end": str(end_str),
            })
    else:
        # Fallback: treat as single span
        spans.append({
            "window": window_name,
            "span": f"span_{str(start_str)[:7]}",
            "start": str(start_str),
            "end": str(end_str),
        })
    return spans


def main():
    parser = argparse.ArgumentParser(description="Compute window/span combinations")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--inventory-marker", required=True, help="Input inventory marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.inventory_marker, "r") as f:
        inv_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    windows = config.get("windows", [])
    scan = config.get("scan", {})
    start = str(scan.get("start", "2022-02-01"))
    end = str(scan.get("end", "2022-03-01"))
    sites = config.get("sites", [])
    splits = config.get("splits", ["train", "test"])

    enabled_windows = [w["name"] for w in windows if w.get("enabled", False)]
    logger.info(f"Enabled windows: {enabled_windows}")
    logger.info(f"Scan range: {start} to {end}")

    all_spans = []
    for window_name in enabled_windows:
        spans = compute_spans(window_name, start, end)
        all_spans.extend(spans)
        logger.info(f"Window '{window_name}': {len(spans)} span(s)")

    logger.info(f"Total: {len(all_spans)} span(s) across {len(enabled_windows)} window(s)")

    marker = {
        "stage": "plan_spans",
        "status": "success",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "enabled_windows": enabled_windows,
        "scan_start": start,
        "scan_end": end,
        "sites": sites,
        "splits": splits,
        "spans": all_spans,
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
