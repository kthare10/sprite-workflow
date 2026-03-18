#!/home/ubuntu/sprite-venv/bin/python3

"""Merge all site snapshots into a central snapshot.

Reads site snapshot markers for all sites and creates a unified central
snapshot directory. The central snapshot aggregates data across all sites
for use by the federated learning server and centralized training.

Corresponds to orchestrator Step F (central part): central_snapshot_window.
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

NORM_TAG = "no-normalization"
PRECIP_TAG = "precip-min:0.0_precip-max:128"
NORM_TAG_COMBINED = f"{NORM_TAG}/{PRECIP_TAG}"
VERSION = "v001"


def main():
    parser = argparse.ArgumentParser(description="Merge site snapshots into central")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--site-marker", action="append", default=[],
                        help="Site snapshot marker file (repeated per site)")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    logger.info(f"Site markers: {args.site_marker}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    site_markers = []
    for marker_path in args.site_marker:
        with open(marker_path, "r") as f:
            site_markers.append(json.load(f))

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    snapshot_root = config.get("paths", {}).get("snapshot_root", "")
    sites = [m.get("site", "unknown") for m in site_markers]

    logger.info(f"Snapshot root: {snapshot_root}")
    logger.info(f"Merging snapshots for sites: {sites}")

    now = datetime.utcnow().isoformat() + "Z"
    central_splits = {}

    # Collect all (window, span_dir, split) combos from site markers
    combos = set()
    for sm in site_markers:
        for split_key in sm.get("splits", {}):
            # split_key like "1mo/span_2022-02/train"
            parts = split_key.split("/")
            if len(parts) == 3:
                combos.add((parts[0], parts[1], parts[2]))

    for window, span_dir_name, split in sorted(combos):
        # Central destination
        central_dir = os.path.join(
            snapshot_root, "central", NORM_TAG, PRECIP_TAG,
            window, VERSION, span_dir_name, split,
        )
        os.makedirs(central_dir, exist_ok=True)

        total_seq_count = 0
        span_key = span_dir_name.replace("span_", "")  # "2022-02"

        for sm in site_markers:
            site = sm.get("site", "unknown")
            split_key = f"{window}/{span_dir_name}/{split}"
            split_info = sm.get("splits", {}).get(split_key, {})
            site_snap_path = split_info.get("path", "")

            if not site_snap_path or not os.path.isdir(site_snap_path):
                site_snap_path = os.path.join(
                    snapshot_root, "sites", site, NORM_TAG, PRECIP_TAG,
                    window, VERSION, span_dir_name, split,
                )

            if not os.path.isdir(site_snap_path):
                logger.warning(f"Site snapshot dir not found: {site_snap_path}")
                continue

            for entry in sorted(os.listdir(site_snap_path)):
                if entry in ("MANIFEST.json", "_SUCCESS") or entry.startswith("."):
                    continue
                src = os.path.join(site_snap_path, entry)
                if not os.path.isdir(src) and not os.path.islink(src):
                    continue

                # Use double-underscore suffix: seq-16-1-0.0094726__KBOX
                dst_name = f"{entry}__{site}"
                dst = os.path.join(central_dir, dst_name)
                if not os.path.exists(dst):
                    real_src = os.path.realpath(src)
                    try:
                        os.symlink(real_src, dst)
                    except OSError:
                        import shutil
                        shutil.copytree(real_src, dst)
                total_seq_count += 1

        # Write central MANIFEST.json (matching reference format)
        manifest = {
            "level": "central",
            "split": split,
            "window": window,
            "span": span_key,
            "version": VERSION,
            "norm_tag": NORM_TAG_COMBINED,
            "seq_count": total_seq_count,
            "sites": sorted(sites),
            "created_at": now,
            "success": True,
        }
        with open(os.path.join(central_dir, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # Write _SUCCESS marker
        with open(os.path.join(central_dir, "_SUCCESS"), "w") as f:
            pass

        combo_key = f"{window}/{span_dir_name}/{split}"
        central_splits[combo_key] = {
            "seq_count": total_seq_count,
            "path": central_dir,
        }

        logger.info(
            f"  central/{window}/{span_dir_name}/{split}: "
            f"{total_seq_count} sequences from {len(sites)} sites"
        )

    marker = {
        "stage": "central_snapshot",
        "status": "success",
        "timestamp": now,
        "snapshot_root": snapshot_root,
        "sites": sites,
        "site_snapshot_count": len(site_markers),
        "central_splits": central_splits,
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
