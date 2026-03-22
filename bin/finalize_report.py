#!/usr/bin/env python3

"""Generate final pipeline report.

Accepts the central snapshot tar and FL configs tar, inspects their
contents, and produces a summary JSON report of the pipeline run.

No SQLite, no absolute paths.
"""

import argparse
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def count_sequences_in_dir(path):
    """Count sequence directories (seq-*) in a directory tree."""
    count = 0
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d.startswith("seq-"):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate final pipeline report")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--central-tar", required=True,
                        help="Input central snapshot tar.gz archive")
    parser.add_argument("--configs-tar", required=True,
                        help="Input FL configs tar.gz archive")
    parser.add_argument("--output", required=True, help="Output report JSON file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    sites = config.get("sites", [])
    now = datetime.utcnow().isoformat() + "Z"

    # Inspect central snapshot tar
    central_tmp = tempfile.mkdtemp(prefix="sprite_report_central_")
    logger.info(f"Extracting {args.central_tar} for inspection")
    with tarfile.open(args.central_tar, "r:gz") as tar:
        tar.extractall(central_tmp)

    # Count sequences and read manifests from central snapshot
    total_central_sequences = 0
    central_splits = {}
    for root, dirs, files in os.walk(central_tmp):
        if "MANIFEST.json" in files:
            manifest_path = os.path.join(root, "MANIFEST.json")
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            rel_path = os.path.relpath(root, central_tmp)
            seq_count = manifest.get("seq_count", 0)
            total_central_sequences += seq_count
            central_splits[rel_path] = {
                "seq_count": seq_count,
                "sites": manifest.get("sites", []),
            }

    shutil.rmtree(central_tmp, ignore_errors=True)

    # Inspect configs tar
    configs_tmp = tempfile.mkdtemp(prefix="sprite_report_configs_")
    logger.info(f"Extracting {args.configs_tar} for inspection")
    with tarfile.open(args.configs_tar, "r:gz") as tar:
        tar.extractall(configs_tmp)

    # Count config runs
    fl_runs = 0
    cen_runs = 0
    for root, dirs, files in os.walk(configs_tmp):
        if "config.yaml" in files:
            rel = os.path.relpath(root, configs_tmp)
            if rel.startswith("fl/"):
                fl_runs += 1
            elif rel.startswith("cen/"):
                cen_runs += 1

    shutil.rmtree(configs_tmp, ignore_errors=True)

    # Determine overall pipeline status
    pipeline_status = "success"
    if total_central_sequences == 0:
        pipeline_status = "empty"

    logger.info(f"Central sequences: {total_central_sequences}")
    logger.info(f"FL runs: {fl_runs}, CEN runs: {cen_runs}")
    logger.info(f"Pipeline status: {pipeline_status}")

    report = {
        "stage": "finalize_report",
        "status": pipeline_status,
        "timestamp": now,
        "sites": sites,
        "central_snapshot": {
            "total_sequences": total_central_sequences,
            "splits": central_splits,
        },
        "training_configs": {
            "fl_runs": fl_runs,
            "cen_runs": cen_runs,
            "total_runs": fl_runs + cen_runs,
        },
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Final report written: {args.output}")


if __name__ == "__main__":
    main()
