#!/home/ubuntu/sprite-venv/bin/python3

"""Create a site snapshot from preprocessed data.

Assembles preprocessed data for a single site into a versioned snapshot
directory with MANIFEST.json and _SUCCESS markers per split. Snapshots
provide a stable, immutable view of training/test data for a given
window and span.

Corresponds to orchestrator Step F (site part): snapshot_site() ->
site_snapshot_window.
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
    parser = argparse.ArgumentParser(description="Create site snapshot")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--site", required=True, help="Site identifier (e.g. KBOX)")
    parser.add_argument("--preproc-marker", required=True, help="Input preproc marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    logger.info(f"Site: {args.site}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.preproc_marker, "r") as f:
        preproc_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    snapshot_root = config.get("paths", {}).get("snapshot_root", "")
    nc_subset_root = config.get("paths", {}).get("nc_subset_root", "")
    splits = config.get("splits", ["train", "test"])
    windows_cfg = config.get("windows", [])

    enabled_windows = [w["name"] for w in windows_cfg if w.get("enabled", False)]

    logger.info(f"Snapshot root: {snapshot_root}")
    logger.info(f"NC subset root: {nc_subset_root}")
    logger.info(f"Splits: {splits}")

    now = datetime.utcnow().isoformat() + "Z"
    snapshot_splits = {}

    frozen_months = preproc_marker.get("frozen_months", [])

    for window in enabled_windows:
        for yyyymm in frozen_months:
            if not yyyymm:
                continue
            # Convert YYYYMM to span label and period_key
            span_key = f"{yyyymm[:4]}-{yyyymm[4:6]}"  # "2022-02"
            span_dir_name = f"span_{span_key}"          # "span_2022-02"

            for split in splits:
                # Source: nc_subset/{SITE}/no-normalization/precip-min:0.0_precip-max:128/{split}/
                src_base = os.path.join(
                    nc_subset_root, args.site, NORM_TAG, PRECIP_TAG, split
                )

                # Destination: snapshots/sites/{SITE}/no-normalization/
                #   precip-min:0.0_precip-max:128/{window}/v001/span_{YYYY-MM}/{split}/
                dst_base = os.path.join(
                    snapshot_root, "sites", args.site, NORM_TAG, PRECIP_TAG,
                    window, VERSION, span_dir_name, split,
                )
                os.makedirs(dst_base, exist_ok=True)

                seq_count = 0

                if os.path.isdir(src_base):
                    for seq_dir_name in sorted(os.listdir(src_base)):
                        if seq_dir_name.startswith("."):
                            continue
                        seq_src = os.path.join(src_base, seq_dir_name)
                        if not os.path.isdir(seq_src):
                            continue

                        seq_dst = os.path.join(dst_base, seq_dir_name)
                        if not os.path.exists(seq_dst):
                            try:
                                os.symlink(
                                    os.path.abspath(seq_src),
                                    seq_dst,
                                )
                            except OSError:
                                import shutil
                                shutil.copytree(seq_src, seq_dst)
                        seq_count += 1

                # Write MANIFEST.json (matching reference format)
                manifest = {
                    "level": "site",
                    "site": args.site,
                    "split": split,
                    "window": window,
                    "span": span_key,
                    "version": VERSION,
                    "norm_tag": NORM_TAG_COMBINED,
                    "seq_count": seq_count,
                    "created_at": now,
                    "success": True,
                }
                with open(os.path.join(dst_base, "MANIFEST.json"), "w") as f:
                    json.dump(manifest, f, indent=2)

                # Write _SUCCESS marker
                with open(os.path.join(dst_base, "_SUCCESS"), "w") as f:
                    pass

                split_key = f"{window}/{span_dir_name}/{split}"
                snapshot_splits[split_key] = {
                    "status": "success",
                    "seq_count": seq_count,
                    "path": dst_base,
                }

                logger.info(
                    f"  {args.site}/{window}/{span_dir_name}/{split}: "
                    f"{seq_count} sequences"
                )

    marker = {
        "stage": "snapshot",
        "status": "success",
        "timestamp": now,
        "site": args.site,
        "snapshot_root": snapshot_root,
        "splits": snapshot_splits,
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
