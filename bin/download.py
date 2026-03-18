#!/home/ubuntu/sprite-venv/bin/python3

"""Download MRMS data from AWS S3.

Reads download_config.yaml for S3 source parameters, date range, station list,
and chunk settings. Downloads .grib2.gz files, decompresses, clips to station
bounding box, converts to NetCDF, and splits into train/test sets.

Produces a JSON marker file summarizing download results.
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
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


def generate_timestamps(day_str, step_minutes=2):
    """Generate all HH:MM:SS timestamps for a given day at the given step."""
    base = datetime.strptime(day_str, "%Y%m%d")
    stamps = []
    t = base
    end = base + timedelta(days=1)
    while t < end:
        stamps.append(t.strftime("%H%M%S"))
        t += timedelta(minutes=step_minutes)
    return stamps


def download_and_convert_file(
    s3_client, bucket, s3_key, station, lat, lon,
    clip_deg, out_path, max_attempts, base_delay
):
    """Download a single .grib2.gz, decompress, clip, and save as NetCDF."""
    import time

    for attempt in range(1, max_attempts + 1):
        try:
            with tempfile.NamedTemporaryFile(suffix=".grib2.gz", delete=False) as tmp_gz:
                tmp_gz_path = tmp_gz.name
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp_grib:
                tmp_grib_path = tmp_grib.name

            # Download from S3
            s3_client.download_file(bucket, s3_key, tmp_gz_path)

            # Decompress gzip
            with gzip.open(tmp_gz_path, "rb") as f_in:
                with open(tmp_grib_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.unlink(tmp_gz_path)

            # Open with xarray/cfgrib and clip
            import gc
            import xarray as xr

            try:
                ds = xr.open_dataset(tmp_grib_path, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
            except Exception:
                ds = xr.open_dataset(tmp_grib_path, engine="cfgrib")

            # Clip to bounding box around station
            lat_min, lat_max = lat - clip_deg, lat + clip_deg

            # Detect coordinate names
            if "latitude" in ds.dims:
                lat_coord, lon_coord = "latitude", "longitude"
            elif "lat" in ds.dims:
                lat_coord, lon_coord = "lat", "lon"
            else:
                lat_coord = next((c for c in ds.coords if "lat" in c.lower()), list(ds.dims)[0])
                lon_coord = next((c for c in ds.coords if "lon" in c.lower()), list(ds.dims)[1])

            # Handle longitude convention (MRMS uses 0-360)
            lon_vals = ds[lon_coord].values
            if lon_vals.max() > 180:
                lon_360 = lon % 360 if lon < 0 else lon + 360 if lon < 0 else lon
                # For negative longitudes, convert to 0-360
                if lon < 0:
                    lon_360 = lon + 360
                else:
                    lon_360 = lon
                lon_min_sel = lon_360 - clip_deg
                lon_max_sel = lon_360 + clip_deg
            else:
                lon_min_sel = lon - clip_deg
                lon_max_sel = lon + clip_deg

            # Try both orderings for latitude (N->S or S->N)
            ds_clipped = ds.sel(
                {lat_coord: slice(lat_min, lat_max),
                 lon_coord: slice(lon_min_sel, lon_max_sel)}
            )
            if any(s == 0 for s in ds_clipped.sizes.values()):
                ds_clipped = ds.sel(
                    {lat_coord: slice(lat_max, lat_min),
                     lon_coord: slice(lon_min_sel, lon_max_sel)}
                )

            # Load into memory, then close source to free grib buffers
            ds_clipped = ds_clipped.load()
            ds.close()
            del ds

            # Save as compressed NetCDF (matching reference format)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            encoding = {}
            for var in ds_clipped.data_vars:
                encoding[var] = {
                    "zlib": True, "complevel": 4, "shuffle": True,
                    "dtype": "float32",
                }
            for coord in ds_clipped.coords:
                if ds_clipped[coord].size > 1:
                    encoding[coord] = {
                        "zlib": True, "complevel": 3, "shuffle": True,
                        "dtype": "float32",
                    }
            ds_clipped.to_netcdf(out_path, encoding=encoding)
            ds_clipped.close()
            del ds_clipped
            gc.collect()

            os.unlink(tmp_grib_path)
            return True

        except Exception as e:
            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed for {s3_key}: {e}"
            )
            # Cleanup temp files
            for p in [tmp_gz_path, tmp_grib_path]:
                if os.path.exists(p):
                    os.unlink(p)
            if attempt < max_attempts:
                import time
                time.sleep(base_delay * (2 ** (attempt - 1)))

    return False


def main():
    parser = argparse.ArgumentParser(description="Download MRMS data from AWS S3")
    parser.add_argument("--config", required=True, help="Path to download_config.yaml")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    source = config.get("source", "s3")
    product = config.get("product", "PrecipRate")
    start_date = str(config.get("start_date", "2022-02-01"))
    end_date = str(config.get("end_date", "2022-03-01"))
    radars = config.get("radars", [])
    stations = config.get("stations", {})
    output_root = config.get("output", {}).get("root", "/tmp/mrms_download")
    raw_store = config.get("output", {}).get("nc_subset", "/tmp/mrms_raw_store")
    chunks = config.get("chunks", 44)
    max_attempts = config.get("retries", {}).get("max_attempts", 3)
    base_delay = config.get("retries", {}).get("base_delay_seconds", 5)
    clip_deg = config.get("clip_degrees", 3)
    split_mode = config.get("split", {}).get("mode", "random")
    split_ratio = config.get("split", {}).get("ratio", 0.2)
    split_seed = config.get("split", {}).get("seed", 2025)
    step_minutes = config.get("audit", {}).get("step_minutes", 2)

    logger.info(f"Source: {source}, Product: {product}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Radars: {radars}")
    logger.info(f"Output root: {output_root}, Raw store: {raw_store}")
    logger.info(f"Chunks: {chunks}, Clip degrees: {clip_deg}")

    # Initialize boto3 S3 client (unsigned for public bucket)
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    s3_client = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
    bucket = "noaa-mrms-pds"

    # Build list of all download tasks
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    tasks = []
    current = dt_start
    while current < dt_end:
        day_str = current.strftime("%Y%m%d")
        timestamps = generate_timestamps(day_str, step_minutes)

        for radar in radars:
            station_info = stations.get(radar, {})
            lat = station_info.get("latitude", 0.0)
            lon = station_info.get("longitude", 0.0)
            region = station_info.get("region", "CONUS")

            for hhmmss in timestamps:
                ts_full = f"{day_str}_{hhmmss}"
                split = ts_to_split(ts_full, ratio=split_ratio, seed=split_seed)
                s3_key = (
                    f"{region}/{product}_00.00/{day_str}/"
                    f"MRMS_{product}_00.00_{day_str}-{hhmmss}.grib2.gz"
                )
                out_path = os.path.join(
                    raw_store, radar, split, day_str, f"{day_str}_{hhmmss}.nc"
                )
                tasks.append((s3_key, radar, lat, lon, out_path, ts_full))

        current += timedelta(days=1)

    logger.info(f"Total download tasks: {len(tasks)}")

    # Execute downloads in parallel
    downloaded_files = 0
    failed_files = 0
    skipped_files = 0

    def do_download(task):
        s3_key, radar, lat, lon, out_path, ts_full = task
        if os.path.exists(out_path):
            return "skipped"
        ok = download_and_convert_file(
            s3_client, bucket, s3_key, radar, lat, lon,
            clip_deg, out_path, max_attempts, base_delay,
        )
        return "ok" if ok else "fail"

    # Limit parallelism to control memory (each grib2 can use ~500MB)
    with ThreadPoolExecutor(max_workers=min(chunks, 4)) as pool:
        futures = {pool.submit(do_download, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result == "ok":
                downloaded_files += 1
            elif result == "fail":
                failed_files += 1
            else:
                skipped_files += 1
            if (i + 1) % 500 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(tasks)} "
                    f"(ok={downloaded_files}, fail={failed_files}, skip={skipped_files})"
                )

    logger.info(
        f"Download complete: {downloaded_files} ok, {failed_files} failed, "
        f"{skipped_files} skipped out of {len(tasks)} total"
    )

    marker = {
        "stage": "download",
        "status": "success" if failed_files == 0 else "partial",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "product": product,
        "start_date": start_date,
        "end_date": end_date,
        "radars": radars,
        "output_root": output_root,
        "nc_subset": raw_store,
        "downloaded_files": downloaded_files,
        "failed_files": failed_files,
        "skipped_files": skipped_files,
        "total_tasks": len(tasks),
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
