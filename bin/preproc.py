#!/usr/bin/env python3

"""Preprocess raw data for a single site.

Absorbs the logic previously split across inventory.py, plan_spans.py,
freeze.py, and the old preproc.py into a single Pegasus-native script.

Accepts a raw tar archive for one site, extracts it, walks .nc files,
computes window/span combinations from config, verifies file completeness,
groups files into fixed-length contiguous sequences, and packages the
result into an output tar archive.

No SQLite, no absolute paths, no hard links.
"""

import argparse
import calendar
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ts_to_split(ts_str, ratio=0.2, seed=2025):
    """Deterministically assign a timestamp to train or test split."""
    h = hashlib.md5(f"{seed}:{ts_str}".encode()).hexdigest()
    return "test" if (int(h, 16) % 1000) < int(ratio * 1000) else "train"


def parse_ts(ts_str):
    """Parse YYYYMMDD_HHMMSS into a datetime."""
    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")


def months_in_range(start_str, end_str):
    """Yield YYYY-MM strings for each month touched by [start, end)."""
    start = datetime.strptime(str(start_str), "%Y-%m-%d")
    end = datetime.strptime(str(end_str), "%Y-%m-%d")
    current = start.replace(day=1)
    while current < end:
        yield current.strftime("%Y-%m")
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def compute_spans(window_name, start_str, end_str):
    """Compute span labels for a given window type and date range."""
    spans = []
    if window_name in ("1mo", "3mo", "6mo", "12mo", "24mo", "48mo"):
        for m in months_in_range(start_str, end_str):
            spans.append({
                "window": window_name,
                "span": f"span_{m}",
                "month": m,
            })
    else:
        for m in months_in_range(start_str, end_str):
            spans.append({
                "window": window_name,
                "span": f"span_{m}",
                "month": m,
            })
    return spans


def expected_timestamps_for_month(year, month, step_minutes=2):
    """Generate all expected YYYYMMDD_HHMMSS timestamps for a month."""
    days_in_month = calendar.monthrange(year, month)[1]
    timestamps = []
    for day in range(1, days_in_month + 1):
        day_str = f"{year:04d}{month:02d}{day:02d}"
        t = datetime(year, month, day)
        end = t + timedelta(days=1)
        while t < end:
            ts = f"{day_str}_{t.strftime('%H%M%S')}"
            timestamps.append(ts)
            t += timedelta(minutes=step_minutes)
    return timestamps


def group_into_sequences(files_with_ts, seq_length=16, gap_minutes=10):
    """Group sorted (ts_str, path) pairs into contiguous sequences.

    Identifies contiguous runs (gap <= gap_minutes), then slices each
    run into fixed-length sequences of seq_length timesteps.
    """
    if not files_with_ts:
        return []

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
    parser = argparse.ArgumentParser(
        description="Preprocess raw data for a single site (inventory + spans + freeze + preproc)"
    )
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--site", required=True, help="Site identifier (e.g. KBOX)")
    parser.add_argument("--raw-tar", required=True, help="Input raw tar.gz archive for site")
    parser.add_argument("--output", required=True, help="Output sequences tar.gz archive")
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
    freeze_cfg = config.get("freeze", {})
    missing_tol = freeze_cfg.get("missing_ratio_tol", {"train": 0.25, "test": 0.85})
    preproc_cfg = config.get("preprocessor", {})
    seq_length = preproc_cfg.get("sequence_length", 16)
    gap_minutes = preproc_cfg.get("gap_minutes", 10)
    norm_tag = preproc_cfg.get("normalization", "no-normalization")
    precip_tag = preproc_cfg.get("precip_range", "precip-min:0.0_precip-max:128")
    step_minutes = 2

    # Extract raw tar to temp directory
    raw_dir = tempfile.mkdtemp(prefix="sprite_raw_")
    logger.info(f"Extracting {args.raw_tar} to {raw_dir}")
    with tarfile.open(args.raw_tar, "r:gz") as tar:
        tar.extractall(raw_dir)

    # Output temp directory for sequences
    seq_dir = tempfile.mkdtemp(prefix="sprite_seq_")

    # Compute spans from config (replaces plan_spans.py)
    all_spans = []
    for window_name in enabled_windows:
        spans = compute_spans(window_name, scan_start, scan_end)
        all_spans.extend(spans)
    logger.info(f"Computed {len(all_spans)} spans across {len(enabled_windows)} windows")

    # Inventory: walk extracted files (replaces inventory.py)
    # Files are organized as {split}/{YYYYMMDD}/{YYYYMMDD_HHMMSS}.nc
    file_index = {}  # {split: {yyyymm: [(ts_str, path), ...]}}
    for split in splits:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        file_index[split] = {}
        for day_dir_name in sorted(os.listdir(split_dir)):
            day_path = os.path.join(split_dir, day_dir_name)
            if not os.path.isdir(day_path):
                continue
            yyyymm = day_dir_name[:6]  # e.g. "202202"
            if yyyymm not in file_index[split]:
                file_index[split][yyyymm] = []
            for nc_file in sorted(os.listdir(day_path)):
                if not nc_file.endswith(".nc"):
                    continue
                ts_str = nc_file.replace(".nc", "")
                nc_path = os.path.join(day_path, nc_file)
                file_index[split][yyyymm].append((ts_str, nc_path))

    total_indexed = sum(
        len(files) for split_data in file_index.values()
        for files in split_data.values()
    )
    logger.info(f"Indexed {total_indexed} files from raw tar")

    processed_files = 0
    total_sequences = 0

    for span_info in all_spans:
        window = span_info["window"]
        month_str = span_info["month"]  # "2022-02"
        yyyymm = month_str.replace("-", "")  # "202202"

        try:
            year = int(month_str.split("-")[0])
            month = int(month_str.split("-")[1])
        except (ValueError, IndexError):
            logger.warning(f"Cannot parse month from span: {span_info}")
            continue

        for split in splits:
            # Freeze check: verify completeness (replaces freeze.py)
            expected_ts = expected_timestamps_for_month(year, month, step_minutes)
            split_expected = [
                ts for ts in expected_ts
                if ts_to_split(ts, ratio=0.2, seed=2025) == split
            ]
            actual_files = file_index.get(split, {}).get(yyyymm, [])
            actual_ts_set = {ts for ts, _ in actual_files}
            missing_count = sum(1 for ts in split_expected if ts not in actual_ts_set)
            expected_count = len(split_expected)
            missing_ratio = missing_count / expected_count if expected_count > 0 else 0.0
            tol = missing_tol.get(split, 0.25)

            logger.info(
                f"  {args.site}/{split}/{yyyymm}: "
                f"E={expected_count} A={len(actual_files)} "
                f"missing={missing_ratio:.2%} (tol={tol})"
            )

            if not actual_files:
                continue

            # Sort and group into sequences (existing preproc logic)
            sorted_files = sorted(actual_files, key=lambda x: x[0])
            sequences = group_into_sequences(sorted_files, seq_length, gap_minutes)

            logger.info(
                f"  {args.site}/{split}/{yyyymm}: "
                f"{len(actual_files)} files -> {len(sequences)} sequences"
            )

            # Write sequences to output dir
            subset_base = os.path.join(seq_dir, norm_tag, precip_tag, split)

            for idx, seq in enumerate(sequences, start=1):
                avg_precip = compute_avg_precip(seq[0][1])
                seq_name = f"seq-{seq_length}-{idx}-{avg_precip:.7g}"

                seq_out_dir = os.path.join(subset_base, seq_name)
                os.makedirs(seq_out_dir, exist_ok=True)

                for ts_str, src_path in seq:
                    dst_path = os.path.join(seq_out_dir, os.path.basename(src_path))
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                    processed_files += 1

                total_sequences += 1

    logger.info(
        f"Preprocessing complete: {processed_files} files in "
        f"{total_sequences} sequences"
    )

    # Package into tar.gz archive
    logger.info(f"Creating output archive: {args.output}")
    with tarfile.open(args.output, "w:gz") as tar:
        for root, dirs, files in os.walk(seq_dir):
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, seq_dir)
                tar.add(full_path, arcname=arcname)

    # Cleanup temp directories
    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(seq_dir, ignore_errors=True)

    logger.info(f"Output archive written: {args.output}")


if __name__ == "__main__":
    main()
