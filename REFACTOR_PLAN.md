# Plan: Pegasus-Native Refactor of sprite-workflow

## Context

The sprite-workflow currently relies on anti-patterns that break Pegasus's sandboxed execution model:
- **Absolute paths** via `__DATA_ROOT__` and `__FL_ROOT__` tokens resolved at DAG generation time
- **Shared SQLite database** (`sprite.sqlite`) passed between jobs via filesystem, not Pegasus staging
- **Hard links and symlinks** (`os.link()`, `os.symlink()`) that assume shared filesystem
- **`--bind /home/ubuntu`** Singularity mount required because scripts write to absolute paths outside the job's working directory
- **Single monolithic download job** that fetches all sites at once

These prevent the workflow from running on any cluster without the specific `/home/ubuntu` mount and shared filesystem. The refactor makes every job self-contained with explicit Pegasus File inputs/outputs.

## Design Principles

1. **Tar archives** replace filesystem directories as the unit of data transfer between jobs
2. **CSV/JSON files** replace SQLite for inter-job metadata
3. **Relative paths only** — no `__DATA_ROOT__`, no absolute paths, no bind mounts
4. **Per-site download parallelism** — one download job per site instead of one monolithic job
5. **Pegasus `infer_dependencies=True`** with explicit `add_inputs()`/`add_outputs()` on File objects

## New DAG Structure

```
For each SITE in [KBOX, KBYX, KENX, KLGX, KTLX, KVNX, PAHG]:
  download_{SITE} ──> preproc_{SITE} ──> snapshot_{SITE} ─┐
                                                            ├──> central_snapshot ──> prepare_configs ──> finalize_report
                                                            │
```

Removed jobs: `inventory`, `plan_spans`, `freeze_{SITE}` (logic absorbed into other scripts)
Renamed jobs: `enqueue_submit` → `prepare_configs`, `poll_retry` removed (no-op in Pegasus mode)

## Files Modified

| File | Change |
|------|--------|
| `workflow_generator.py` | Complete rewrite: per-site download, tar-based File objects, no bind mount, no DATA_ROOT |
| `bin/download.py` | Refactor: single-site mode, output tar archive instead of writing to absolute paths |
| `bin/inventory.py` | **Delete** — logic merged into `preproc.py` |
| `bin/plan_spans.py` | **Delete** — logic merged into `preproc.py` |
| `bin/freeze.py` | **Delete** — logic merged into `preproc.py` |
| `bin/preproc.py` | Refactor: accept site tar input, compute spans inline, output sequences tar |
| `bin/snapshot.py` | Refactor: accept sequences tar, output snapshot tar (no symlinks) |
| `bin/central_snapshot.py` | Refactor: accept per-site snapshot tars, merge into central tar |
| `bin/enqueue_submit.py` | Refactor → `bin/prepare_configs.py`: accept central tar, output FL config files with relative paths |
| `bin/poll_retry.py` | **Delete** — no-op in Pegasus mode |
| `bin/finalize_report.py` | Refactor: accept config files + central tar, generate report JSON |
| `experiment_config.yaml` | Remove `paths:` section entirely, remove `__DATA_ROOT__`/`__FL_ROOT__` tokens |
| `download_config.yaml` | Remove `__DATA_ROOT__` tokens from output paths, keep station/S3 config |

## Detailed Script Changes

### `bin/download.py` — Per-site tar output

**Current**: Downloads ALL sites into `{DATA_ROOT}/raw_store/{SITE}/...`, writes marker JSON.
**New**: Accept `--site SITE` argument. Download only that site. Write output to `{site}_raw.tar.gz`.

```
Arguments: --config download_config.yaml --site KBOX --output kbox_raw.tar.gz
Input:     download_config.yaml (Pegasus File)
Output:    kbox_raw.tar.gz (Pegasus File)
```

Key changes:
- Add `--site` and `--output` CLI args
- Download to a local temp directory (relative `./download_tmp/`)
- `tar czf` the downloaded .nc files at the end
- Remove all `__DATA_ROOT__` path resolution — read station config and S3 paths from download_config.yaml directly

### `bin/preproc.py` — Absorbs inventory + plan_spans + freeze logic

**Current**: Reads SQLite `raw_files` table, creates hard links in `nc_subset/`, writes `seq_index` table.
**New**: Accept site tar, compute everything inline, output sequences tar.

```
Arguments: --config experiment_config.yaml --site KBOX --raw-tar kbox_raw.tar.gz --output kbox_sequences.tar.gz
Input:     experiment_config.yaml, kbox_raw.tar.gz (Pegasus Files)
Output:    kbox_sequences.tar.gz (Pegasus File)
```

Key changes:
- Extract tar to local temp dir
- Walk extracted .nc files (replaces `inventory.py` SQLite indexing)
- Compute window/span combinations from config (replaces `plan_spans.py`)
- Verify file completeness per month (replaces `freeze.py` audit)
- Group files into 16-timestep sequences (existing preproc logic)
- Copy files into sequence dirs (replaces hard links)
- `tar czf` the sequence directory tree
- No SQLite, no absolute paths

### `bin/snapshot.py` — Tar-to-tar, no symlinks

**Current**: Creates symlinks from snapshot dir to nc_subset sequences.
**New**: Accept sequences tar, repack into snapshot tar with proper directory structure.

```
Arguments: --config experiment_config.yaml --site KBOX --sequences-tar kbox_sequences.tar.gz --output kbox_snapshot.tar.gz
Input:     experiment_config.yaml, kbox_sequences.tar.gz (Pegasus Files)
Output:    kbox_snapshot.tar.gz (Pegasus File)
```

Key changes:
- Extract sequences tar to temp dir
- Organize into `{window}/v001/span_{YYYY-MM}/{split}/` structure
- Copy actual files (not symlinks)
- `tar czf` the organized tree

### `bin/central_snapshot.py` — Merge per-site tars

**Current**: Merges site snapshots via symlinks with `__{SITE}` suffixes using `os.path.realpath()`.
**New**: Extract all site tars, merge with `__{SITE}` suffixes, output central tar.

```
Arguments: --config experiment_config.yaml --site-tars kbox_snapshot.tar.gz kbyx_snapshot.tar.gz ... --sites KBOX KBYX ... --output central_snapshot.tar.gz
Input:     experiment_config.yaml, all site snapshot tars (Pegasus Files)
Output:    central_snapshot.tar.gz (Pegasus File)
```

Key changes:
- Extract each site tar
- Copy files with `__{SITE}` suffix appended to sequence directory names
- Merge all into single directory tree
- `tar czf` the merged tree

### `bin/prepare_configs.py` (renamed from `enqueue_submit.py`) — Relative-path FL configs

**Current**: Writes FL training configs with absolute paths, writes to SQLite `jobs` table.
**New**: Accept central tar, generate config files with relative paths only.

```
Arguments: --config experiment_config.yaml --central-tar central_snapshot.tar.gz --output fl_configs.tar.gz
Input:     experiment_config.yaml, central_snapshot.tar.gz (Pegasus Files)
Output:    fl_configs.tar.gz (Pegasus File) — contains config.yaml, clients.map.json, flwr.override.toml per run
```

Key changes:
- Extract central tar to inspect available windows/spans
- Generate `config.yaml` with relative data paths (e.g., `data/{window}/v001/span_{date}/{split}/`)
- Generate `clients.map.json` mapping site names to relative directories
- No SQLite, no `_RUNNING` markers, no absolute paths
- Package all config files into output tar

### `bin/finalize_report.py` — Tar input

**Current**: Reads markers and directory tree with absolute paths.
**New**: Accept config tar + central tar, summarize pipeline.

```
Arguments: --config experiment_config.yaml --central-tar central_snapshot.tar.gz --configs-tar fl_configs.tar.gz --output final_report.json
Input:     experiment_config.yaml, central_snapshot.tar.gz, fl_configs.tar.gz (Pegasus Files)
Output:    final_report.json (Pegasus File)
```

### `experiment_config.yaml` — No paths section

Remove the entire `paths:` block. Remove `__DATA_ROOT__` and `__FL_ROOT__` tokens from `fl:`, `centralized:`, and `preprocessor:` sections. Keep: `sites`, `splits`, `windows`, `scan`, `freeze` (thresholds only), `preprocessor` (sequence_length, normalization params).

### `download_config.yaml` — No absolute paths

Remove `__DATA_ROOT__` from `output:` and `audit:` sections. Keep: `stations`, `radars`, `start_date`, `end_date`, S3 source config. The output paths are now implicit (local temp dirs inside each script).

## `workflow_generator.py` — Complete Rewrite

```python
# Key structure (pseudocode):

container = Container(
    "sprite_fl_container",
    container_type=Container.SINGULARITY,
    image="docker://kthare10/sprite-fl:latest",
    image_site="docker_hub",
)
# NO --bind /home/ubuntu, NO DATA_ROOT env var

# Register config files in Replica Catalog
experiment_config = File("experiment_config.yaml")
download_config = File("download_config.yaml")
rc.add_replica("local", experiment_config, os.path.join(base_dir, "experiment_config.yaml"))
rc.add_replica("local", download_config, os.path.join(base_dir, "download_config.yaml"))

# Per-site fan-out
for site in sites:
    site_lower = site.lower()
    raw_tar = File(f"{site_lower}_raw.tar.gz")
    seq_tar = File(f"{site_lower}_sequences.tar.gz")
    snap_tar = File(f"{site_lower}_snapshot.tar.gz")

    download_job = Job("download") \
        .add_args("--config", download_config, "--site", site, "--output", raw_tar) \
        .add_inputs(download_config) \
        .add_outputs(raw_tar, stage_out=False)

    preproc_job = Job("preproc") \
        .add_args("--config", experiment_config, "--site", site,
                  "--raw-tar", raw_tar, "--output", seq_tar) \
        .add_inputs(experiment_config, raw_tar) \
        .add_outputs(seq_tar, stage_out=False)

    snapshot_job = Job("snapshot") \
        .add_args("--config", experiment_config, "--site", site,
                  "--sequences-tar", seq_tar, "--output", snap_tar) \
        .add_inputs(experiment_config, seq_tar) \
        .add_outputs(snap_tar, stage_out=False)

    site_snapshot_tars.append(snap_tar)
    wf.add_jobs(download_job, preproc_job, snapshot_job)

# Fan-in: central snapshot
central_tar = File("central_snapshot.tar.gz")
central_job = Job("central_snapshot") \
    .add_args("--config", experiment_config,
              "--site-tars", *site_snapshot_tars,
              "--sites", *sites,
              "--output", central_tar) \
    .add_inputs(experiment_config, *site_snapshot_tars) \
    .add_outputs(central_tar, stage_out=False)

# Prepare FL configs
configs_tar = File("fl_configs.tar.gz")
prepare_job = Job("prepare_configs") \
    .add_args("--config", experiment_config,
              "--central-tar", central_tar, "--output", configs_tar) \
    .add_inputs(experiment_config, central_tar) \
    .add_outputs(configs_tar, stage_out=False)

# Final report
report = File("final_report.json")
finalize_job = Job("finalize_report") \
    .add_args("--config", experiment_config,
              "--central-tar", central_tar, "--configs-tar", configs_tar,
              "--output", report) \
    .add_inputs(experiment_config, central_tar, configs_tar) \
    .add_outputs(report, stage_out=True)

wf.add_jobs(central_job, prepare_job, finalize_job)
```

## Deleted Files

- `bin/inventory.py` — logic absorbed into `preproc.py`
- `bin/plan_spans.py` — logic absorbed into `preproc.py`
- `bin/freeze.py` — logic absorbed into `preproc.py`
- `bin/poll_retry.py` — no-op in Pegasus mode

## Verification

1. **Generate workflow**: `python workflow_generator.py --config experiment_config.yaml -o workflow.yml`
2. **Inspect DAG**: Verify per-site fan-out, tar File objects, no `--bind`, no `DATA_ROOT`
3. **Inspect generated YAML**: Confirm all jobs use relative paths and tar inputs/outputs
4. **Test on submit host**: `ssh pegasus`, pull changes, run `pegasus-plan --submit -s condorpool -o local workflow.yml`
5. **Verify container**: No `--bind /home/ubuntu` in Singularity args, no `DATA_ROOT` env var
6. **Check job logs**: Confirm scripts extract tars locally and produce output tars
