#!/usr/bin/env python3

"""Generate final pipeline report.

Aggregates results from all pipeline stages into a summary report.
Queries sprite.sqlite for job completion, data quality (freezes),
and sequence counts (seq_index).
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate final pipeline report")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--poll-retry-marker", required=True,
                        help="Input poll/retry marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.poll_retry_marker, "r") as f:
        pr_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sites = config.get("sites", [])
    db_dir = config.get("paths", {}).get("db_dir", "")
    db_path = os.path.join(db_dir, "sprite.sqlite") if db_dir else "sprite.sqlite"

    completed_jobs = pr_marker.get("completed_jobs", [])
    failed_jobs = pr_marker.get("failed_jobs", [])

    now = datetime.utcnow().isoformat() + "Z"

    # Query database for comprehensive summary
    report = {
        "jobs": {"total": 0, "done": 0, "failed": 0, "queued": 0},
        "freezes": {"total": 0, "success": 0, "partial": 0},
        "raw_files": {"total": 0, "per_site": {}},
        "sequences": {"total": 0, "per_site": {}},
    }

    try:
        conn = sqlite3.connect(db_path, timeout=120)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=120000")

        # Jobs summary
        for row in conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ):
            status, count = row
            report["jobs"]["total"] += count
            if status in report["jobs"]:
                report["jobs"][status] = count

        # Freezes summary
        for row in conn.execute(
            "SELECT status, COUNT(*) FROM freezes GROUP BY status"
        ):
            status, count = row
            report["freezes"]["total"] += count
            status_key = status.lower().replace("success", "success").replace("partial", "partial")
            if status_key in ("success", "partial"):
                report["freezes"][status_key] = count

        # Raw files summary
        for row in conn.execute(
            "SELECT site, COUNT(*) FROM raw_files GROUP BY site"
        ):
            site, count = row
            report["raw_files"]["per_site"][site] = count
            report["raw_files"]["total"] += count

        # Sequences summary
        for row in conn.execute(
            "SELECT site, COUNT(*) FROM seq_index GROUP BY site"
        ):
            site, count = row
            report["sequences"]["per_site"][site] = count
            report["sequences"]["total"] += count

        conn.close()

    except Exception as e:
        logger.warning(f"Could not query database: {e}")

    # Determine overall pipeline status
    pipeline_status = "success"
    if failed_jobs:
        pipeline_status = "partial"
    if report["jobs"]["done"] == 0 and report["jobs"]["total"] > 0:
        pipeline_status = "failed"

    logger.info(f"Jobs: {report['jobs']}")
    logger.info(f"Freezes: {report['freezes']}")
    logger.info(f"Raw files: {report['raw_files']['total']}")
    logger.info(f"Sequences: {report['sequences']['total']}")
    logger.info(f"Pipeline status: {pipeline_status}")

    marker = {
        "stage": "finalize_report",
        "status": "success",
        "timestamp": now,
        "sites": sites,
        "total_completed_jobs": len(completed_jobs),
        "total_failed_jobs": len(failed_jobs),
        "pipeline_status": pipeline_status,
        "report": report,
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Final report written: {args.output_marker}")


if __name__ == "__main__":
    main()
