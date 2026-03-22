#!/usr/bin/env python3

"""Create a site snapshot from preprocessed data.

Accepts a sequences tar archive for a single site, extracts it, organizes
the data into a versioned snapshot directory structure
({window}/v001/span_{YYYY-MM}/{split}/), and packages the result into
an output tar archive.

No symlinks, no absolute paths.
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

VERSION = "v001"


def main():
    parser = argparse.ArgumentParser(description="Create site snapshot from sequences tar")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--site", required=True, help="Site identifier (e.g. KBOX)")
    parser.add_argument("--sequences-tar", required=True, help="Input sequences tar.gz archive")
    parser.add_argument("--output", required=True, help="Output snapshot tar.gz archive")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}, Site: {args.site}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    splits = config.get("splits", ["train", "test"])
    windows_cfg = config.get("windows", [])
    enabled_windows = [w["name"] for w in windows_cfg if w.get("enabled", False)]
    scan = config.get("scan", {})
    scan_start = str(scan.get("start", "2022-02-01"))
    scan_end = str(scan.get("end", "2022-03-01"))
    preproc_cfg = config.get("preprocessor", {})
    norm_tag = preproc_cfg.get("normalization", "no-normalization")
    precip_tag = preproc_cfg.get("precip_range", "precip-min:0.0_precip-max:128")

    # Compute months from scan range
    from datetime import datetime as dt
    start_dt = dt.strptime(scan_start, "%Y-%m-%d")
    end_dt = dt.strptime(scan_end, "%Y-%m-%d")
    months = []
    current = start_dt.replace(day=1)
    while current < end_dt:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # Extract sequences tar to temp directory
    seq_dir = tempfile.mkdtemp(prefix="sprite_seq_snap_")
    logger.info(f"Extracting {args.sequences_tar} to {seq_dir}")
    with tarfile.open(args.sequences_tar, "r:gz") as tar:
        tar.extractall(seq_dir)

    # Output temp directory for snapshot
    snap_dir = tempfile.mkdtemp(prefix="sprite_snapshot_")
    now = datetime.utcnow().isoformat() + "Z"

    for window in enabled_windows:
        for month_str in months:
            span_dir_name = f"span_{month_str}"

            for split in splits:
                # Source: {norm_tag}/{precip_tag}/{split}/ (from sequences tar)
                src_base = os.path.join(seq_dir, norm_tag, precip_tag, split)

                # Destination: {window}/v001/span_{YYYY-MM}/{split}/
                dst_base = os.path.join(
                    snap_dir, window, VERSION, span_dir_name, split
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
                            shutil.copytree(seq_src, seq_dst)
                        seq_count += 1

                # Write MANIFEST.json
                manifest = {
                    "level": "site",
                    "site": args.site,
                    "split": split,
                    "window": window,
                    "span": month_str,
                    "version": VERSION,
                    "norm_tag": f"{norm_tag}/{precip_tag}",
                    "seq_count": seq_count,
                    "created_at": now,
                    "success": True,
                }
                with open(os.path.join(dst_base, "MANIFEST.json"), "w") as f:
                    json.dump(manifest, f, indent=2)

                logger.info(
                    f"  {args.site}/{window}/{span_dir_name}/{split}: "
                    f"{seq_count} sequences"
                )

    # Package into tar.gz archive
    logger.info(f"Creating output archive: {args.output}")
    with tarfile.open(args.output, "w:gz") as tar:
        for root, dirs, files in os.walk(snap_dir):
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, snap_dir)
                tar.add(full_path, arcname=arcname)

    # Cleanup temp directories
    shutil.rmtree(seq_dir, ignore_errors=True)
    shutil.rmtree(snap_dir, ignore_errors=True)

    logger.info(f"Output archive written: {args.output}")


if __name__ == "__main__":
    main()
