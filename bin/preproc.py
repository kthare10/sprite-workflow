#!/usr/bin/env python3

"""Preprocess frozen data for a single site.

Groups raw .nc files into contiguous temporal sequences of fixed length,
copies them into the nc_subset directory structure, and records sequences
in the seq_index table.

Corresponds to orchestrator Step E: trigger_preproc() ->
run_preprocessor_for_frozen_month_site.
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NORM_TAG = "no-normalization"
PRECIP_TAG = "precip-min:0.0_precip-max:128"
SEQ_LENGTH = 16  # Fixed sequence length matching reference


def parse_ts(ts_str):
    """Parse YYYYMMDD_HHMMSS into a datetime."""
    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")


def group_into_sequences(files_with_ts, seq_length=SEQ_LENGTH, gap_minutes=10):
    """Group sorted (ts_str, path) pairs into contiguous sequences.

    First identifies contiguous runs (gap <= gap_minutes), then slices
    each run into fixed-length sequences of seq_length timesteps.
    """
    if not files_with_ts:
        return []

    # Step 1: Identify contiguous runs
    runs = []
    current_run = [files_with_ts[0]]
    for i in range(1, len(files_with_ts)):
        prev_dt = parse_ts(files_with_ts[i - 1][0])
        curr_dt = parse_ts(files_with_ts[i][0])
        gap = (curr_dt - prev_dt).total_seconds() / 60.0
        if gap > gap_minutes:
            runs.append(current_run)
            current_run = [files_with_ts[i]]
        else:
            current_run.append(files_with_ts[i])
    if current_run:
        runs.append(current_run)

    # Step 2: Slice runs into fixed-length sequences
    sequences = []
    for run in runs:
        for start in range(0, len(run) - seq_length + 1, seq_length):
            seq = run[start:start + seq_length]
            if len(seq) == seq_length:
                sequences.append(seq)

    return sequences


def compute_avg_precip(nc_path):
    """Compute mean precipitation from a .nc file. Returns 0.0 on failure."""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        for var in ds.data_vars:
            val = float(ds[var].mean().values)
            ds.close()
            return max(0.0, val)
        ds.close()
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Preprocess frozen data for a site")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--site", required=True, help="Site identifier (e.g. KBOX)")
    parser.add_argument("--freeze-marker", required=True, help="Input freeze marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    logger.info(f"Site: {args.site}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.freeze_marker, "r") as f:
        freeze_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    raw_root = config.get("paths", {}).get("raw_root", "")
    nc_subset_root = config.get("paths", {}).get("nc_subset_root", "")
    db_dir = config.get("paths", {}).get("db_dir", "")
    db_path = os.path.join(db_dir, "sprite.sqlite") if db_dir else "sprite.sqlite"
    splits = config.get("splits", ["train", "test"])

    frozen_months = freeze_marker.get("frozen_months", [])

    logger.info(f"NC subset root: {nc_subset_root}")
    logger.info(f"Frozen months: {len(frozen_months)}")

    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    now = datetime.utcnow().isoformat() + "Z"
    processed_files = 0
    total_sequences = 0

    manifest_base = os.path.join(db_dir, "preproc_manifests") if db_dir else "preproc_manifests"

    for fm in frozen_months:
        yyyymm = fm.get("month", "")
        window = fm.get("window", "1mo")

        if not yyyymm:
            continue

        for split in splits:
            # Collect all .nc files for this site/split/month from raw_store
            files_with_ts = []
            year_month_prefix = yyyymm  # e.g. "202202"

            site_split_dir = os.path.join(raw_root, args.site, split)
            if not os.path.isdir(site_split_dir):
                logger.warning(f"Directory not found: {site_split_dir}")
                continue

            for day_dir in sorted(os.listdir(site_split_dir)):
                if not day_dir.startswith(year_month_prefix):
                    continue
                day_path = os.path.join(site_split_dir, day_dir)
                if not os.path.isdir(day_path):
                    continue

                for nc_file in sorted(os.listdir(day_path)):
                    if not nc_file.endswith(".nc"):
                        continue
                    ts_str = nc_file.replace(".nc", "")
                    nc_path = os.path.join(day_path, nc_file)
                    files_with_ts.append((ts_str, nc_path))

            if not files_with_ts:
                logger.info(f"No files for {args.site}/{split}/{yyyymm}")
                continue

            files_with_ts.sort(key=lambda x: x[0])

            # Group into fixed-length contiguous sequences
            sequences = group_into_sequences(files_with_ts, SEQ_LENGTH)
            logger.info(
                f"  {args.site}/{split}/{yyyymm}: "
                f"{len(files_with_ts)} files -> {len(sequences)} sequences"
            )

            # Write sequences to nc_subset (1-based indexing)
            subset_base = os.path.join(
                nc_subset_root, args.site, NORM_TAG, PRECIP_TAG, split
            )

            for idx, seq in enumerate(sequences, start=1):
                # Compute average precip metric from first file
                avg_precip = compute_avg_precip(seq[0][1])
                seq_name = f"seq-{SEQ_LENGTH}-{idx}-{avg_precip:.7g}"

                seq_dir = os.path.join(subset_base, seq_name)
                os.makedirs(seq_dir, exist_ok=True)

                tmin_dt = parse_ts(seq[0][0])
                tmax_dt = parse_ts(seq[-1][0])
                tmin_iso = tmin_dt.strftime("%Y-%m-%dT%H:%M:%S")
                tmax_iso = tmax_dt.strftime("%Y-%m-%dT%H:%M:%S")

                for ts_str, src_path in seq:
                    dst_path = os.path.join(seq_dir, os.path.basename(src_path))
                    if not os.path.exists(dst_path):
                        try:
                            os.link(src_path, dst_path)
                        except OSError:
                            shutil.copy2(src_path, dst_path)
                    processed_files += 1

                # Insert into seq_index
                conn.execute(
                    """INSERT OR REPLACE INTO seq_index
                       (site, split, seq_name, tmin, tmax)
                       VALUES (?,?,?,?,?)""",
                    (args.site, split, seq_name, tmin_iso, tmax_iso),
                )

                total_sequences += 1

            # Write .seq_count file
            os.makedirs(subset_base, exist_ok=True)
            with open(os.path.join(subset_base, ".seq_count"), "w") as f:
                f.write(str(len(sequences)))

        # Write preproc manifest marker
        manifest_dir = os.path.join(manifest_base, args.site, yyyymm)
        os.makedirs(manifest_dir, exist_ok=True)
        with open(os.path.join(manifest_dir, "_SUCCESS"), "w") as f:
            pass

    conn.commit()
    conn.close()

    logger.info(
        f"Preprocessing complete: {processed_files} files in "
        f"{total_sequences} sequences"
    )

    marker = {
        "stage": "preproc",
        "status": "success",
        "timestamp": now,
        "site": args.site,
        "nc_subset_root": nc_subset_root,
        "preprocessor_enabled": True,
        "processed_files": processed_files,
        "total_sequences": total_sequences,
        "frozen_months": [fm.get("month", "") for fm in frozen_months],
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
