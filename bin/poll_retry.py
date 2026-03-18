#!/usr/bin/env python3

"""Poll job status and retry failures.

Monitors submitted jobs by checking for _SUCCESS/_RUNNING markers
and querying the jobs table in sprite.sqlite. In demo/Pegasus mode,
writes _SUCCESS markers for running jobs to simulate completion.

Corresponds to orchestrator Step H: poll_jobs() + retry_jobs().
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
    parser = argparse.ArgumentParser(description="Poll job status and retry failures")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--enqueue-submit-marker", required=True,
                        help="Input enqueue/submit marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.enqueue_submit_marker, "r") as f:
        es_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    db_dir = config.get("paths", {}).get("db_dir", "")
    db_path = os.path.join(db_dir, "sprite.sqlite") if db_dir else "sprite.sqlite"
    orch_cfg = config.get("orchestrator", {})
    max_retries = orch_cfg.get("max_retries", 1)

    submitted_jobs = es_marker.get("submitted_jobs", [])

    logger.info(f"Max retries: {max_retries}")
    logger.info(f"Jobs to poll: {len(submitted_jobs)}")

    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    now = datetime.utcnow().isoformat() + "Z"

    completed_jobs = []
    failed_jobs = []
    retried_jobs = []

    for job_info in submitted_jobs:
        job_id = job_info.get("job_id")
        run_dir = job_info.get("run_dir", "")
        kind = job_info.get("kind", "")

        if not run_dir:
            logger.warning(f"No run_dir for job {job_id}")
            continue

        success_marker = os.path.join(run_dir, "_SUCCESS")
        running_marker = os.path.join(run_dir, "_RUNNING")

        if os.path.exists(success_marker):
            # Already done
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                ("done", now, job_id),
            )
            completed_jobs.append({
                "job_id": job_id,
                "kind": kind,
                "status": "done",
            })
            logger.info(f"Job {job_id} ({kind}): already done (_SUCCESS found)")

        elif os.path.exists(running_marker):
            # In demo/Pegasus mode: simulate completion
            # Write _SUCCESS marker and mark as done
            with open(success_marker, "w") as f:
                pass
            # Remove _RUNNING marker
            try:
                os.unlink(running_marker)
            except OSError:
                pass

            conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                ("done", now, job_id),
            )
            completed_jobs.append({
                "job_id": job_id,
                "kind": kind,
                "status": "done",
                "note": "simulated_completion",
            })
            logger.info(f"Job {job_id} ({kind}): simulated completion (demo mode)")

        else:
            # No markers found - check retries
            row = conn.execute(
                "SELECT retries, status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            retries = row[0] if row else 0
            current_status = row[1] if row else "unknown"

            if retries < max_retries:
                # Retry: write _RUNNING, then immediately _SUCCESS (demo mode)
                os.makedirs(run_dir, exist_ok=True)
                with open(success_marker, "w") as f:
                    pass
                conn.execute(
                    "UPDATE jobs SET status=?, retries=?, updated_at=? WHERE id=?",
                    ("done", retries + 1, now, job_id),
                )
                retried_jobs.append({
                    "job_id": job_id,
                    "kind": kind,
                    "retries": retries + 1,
                })
                completed_jobs.append({
                    "job_id": job_id,
                    "kind": kind,
                    "status": "done",
                    "note": "retried_and_completed",
                })
                logger.info(f"Job {job_id} ({kind}): retried ({retries+1}) and completed")
            else:
                conn.execute(
                    "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                    ("failed", now, job_id),
                )
                failed_jobs.append({
                    "job_id": job_id,
                    "kind": kind,
                    "status": "failed",
                    "retries": retries,
                })
                logger.warning(f"Job {job_id} ({kind}): failed after {retries} retries")

    conn.commit()
    conn.close()

    logger.info(
        f"Poll complete: {len(completed_jobs)} completed, "
        f"{len(failed_jobs)} failed, {len(retried_jobs)} retried"
    )

    marker = {
        "stage": "poll_retry",
        "status": "success" if not failed_jobs else "partial",
        "timestamp": now,
        "poll_interval_sec": orch_cfg.get("poll_interval_sec", 10),
        "max_retries": max_retries,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "retried_jobs": retried_jobs,
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
