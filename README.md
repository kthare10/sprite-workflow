# SPRITE Federated Learning Workflow

Pegasus WMS workflow for the SPRITE project's federated learning pipeline on MRMS radar precipitation data.

## Pipeline Overview

```
download → inventory → plan_spans ─┬─> freeze_{site} → preproc_{site} → snapshot_{site} ──┐
                                    ├─> ...  (parallel per site)                            ├→ central_snapshot
                                    └─> freeze_{site} → preproc_{site} → snapshot_{site} ──┘        │
                                                                                             enqueue_submit
                                                                                                  │
                                                                                             poll_retry
                                                                                                  │
                                                                                          finalize_report
```

### Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Download | `bin/download.py` | Download MRMS data from AWS S3 |
| 2. Inventory | `bin/inventory.py` | Index raw .nc files into SQLite |
| 3. Plan Spans | `bin/plan_spans.py` | Compute window/span combinations from config |
| 4. Freeze | `bin/freeze.py` | Audit & freeze monthly data (per-site, parallel) |
| 5. Preprocess | `bin/preproc.py` | Preprocess frozen data (per-site, parallel) |
| 6a. Snapshot | `bin/snapshot.py` | Create site snapshot (per-site, parallel) |
| 6b. Central Snapshot | `bin/central_snapshot.py` | Merge all site snapshots |
| 7. Enqueue/Submit | `bin/enqueue_submit.py` | Write run configs + submit FL/CEN training jobs |
| 8. Poll/Retry | `bin/poll_retry.py` | Poll job status, retry failures |
| 9. Finalize | `bin/finalize_report.py` | Generate final pipeline report |

## Configuration

Two YAML config files drive the workflow:

- **`download_config.yaml`** — S3 source, product, date range, station list, chunking
- **`experiment_config.yaml`** — paths, sites, windows, freeze settings, FL/centralized training params, Slurm settings

Both configs use **placeholder tokens** instead of hardcoded paths:

| Token | Replaced by | CLI flag | Env var fallback |
|-------|-------------|----------|-----------------|
| `__DATA_ROOT__` | Data storage root | `--data-root` | `$DATA_ROOT` |
| `__FL_ROOT__` | Federated-learning code root | `--fl-root` | `$FL_ROOT` |

At DAG generation time, `workflow_generator.py` reads the template configs, substitutes the placeholders with the resolved paths, and writes the resolved copies to `scratch/resolved_configs/`. The resolved configs are registered with Pegasus and passed to container jobs. `DATA_ROOT` and `FL_ROOT` are also injected as environment variables into every container job.

Sites are read from `experiment_config.yaml` at DAG generation time to create parallel per-site branches.

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate the DAG

```bash
# Minimal — uses $DATA_ROOT / $FL_ROOT env vars (or built-in defaults)
python workflow_generator.py \
    --download-config download_config.yaml \
    --experiment-config experiment_config.yaml \
    --output workflow.yml

# Explicit roots
python workflow_generator.py \
    --download-config download_config.yaml \
    --experiment-config experiment_config.yaml \
    --data-root /data/MRMS/S3_V2 \
    --fl-root /opt/SPRITE \
    --output workflow.yml
```

### Submit the workflow

```bash
pegasus-plan --submit -s condorpool -o local workflow.yml
pegasus-status <run-dir>
```

### CLI options

```
-s, --skip-sites-catalog     Skip site catalog creation
-e, --execution-site-name    Execution site name (default: condorpool)
-o, --output                 Output file (default: workflow.yml)
--download-config            Path to download_config.yaml (required)
--experiment-config          Path to experiment_config.yaml (required)
--data-root PATH             Data storage root (default: $DATA_ROOT or /home/ubuntu/data/MRMS/S3_V2)
--fl-root PATH               Federated-learning code root (default: $FL_ROOT)
```

## Container

The workflow uses a Singularity container pulled from `docker://kthare10/sprite-fl:latest`.

To build locally:

```bash
docker build -t kthare10/sprite-fl:latest -f Docker/Dockerfile .
docker push kthare10/sprite-fl:latest
```

## Architecture

- **Marker-file dependencies**: Each job produces a JSON marker file. Pegasus `infer_dependencies=True` auto-wires the DAG from File object matching.
- **Config-driven fan-out**: `workflow_generator.py` reads `experiment_config.yaml` at generation time to determine the site list and creates N parallel branches.
- **SQLite on shared filesystem**: All jobs access `sprite.sqlite` at a fixed path. Per-site jobs write non-overlapping rows. WAL mode enabled for concurrent access.
- **Stub scripts**: Each `bin/` script has a clear interface (argparse CLI, JSON marker output) with stub implementations. Replace stubs with calls to existing SPRITE library code.
