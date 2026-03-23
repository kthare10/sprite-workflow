# SPRITE Federated Learning Workflow

Pegasus WMS workflow for the SPRITE project's federated learning pipeline on MRMS radar precipitation data.

## Pipeline Overview

```
For each site (KBOX, KBYX, KENX, KLGX, KTLX, KVNX, PAHG):

  download_{site} ──> preproc_{site} ──> snapshot_{site} ─┐
  download_{site} ──> preproc_{site} ──> snapshot_{site} ──┤
  ...               (parallel per site)                    │
  download_{site} ──> preproc_{site} ──> snapshot_{site} ──┘
                                                           │
                                                    central_snapshot
                                                           │
                                                    prepare_configs
                                                      │         │
                                              model_compare_run  ├──> finalize_report
                                                      │         │
                                               visual_compare    └──> visualize (pipeline report)
```

All inter-job data flows through **tar archives** as explicit Pegasus `File` objects. No shared filesystem, no SQLite, no absolute paths.

### Stages

| Stage | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 1. Download | `bin/download.py` | `download_config.yaml` | `{site}_raw.tar.gz` | Download MRMS data from AWS S3 for a single site |
| 2. Preprocess | `bin/preproc.py` | `experiment_config.yaml`, raw tar | `{site}_sequences.tar.gz` | Index files, compute spans, verify completeness, group into sequences |
| 3. Snapshot | `bin/snapshot.py` | `experiment_config.yaml`, sequences tar | `{site}_snapshot.tar.gz` | Organize sequences into versioned window/span/split structure |
| 4. Central Snapshot | `bin/central_snapshot.py` | `experiment_config.yaml`, all site snapshot tars | `central_snapshot.tar.gz` | Merge all site snapshots with `__{SITE}` suffixes |
| 5. Prepare Configs | `bin/prepare_configs.py` | `experiment_config.yaml`, central tar | `fl_configs.tar.gz` | Generate FL/centralized training configs with relative paths |
| 6. Model Compare | `bin/model_compare_run.py` | `experiment_config.yaml`, configs tar, central tar | `model_comparison.json` | Compare FL vs centralized training results |
| 7. Visual Compare | `bin/visual_compare.py` | `experiment_config.yaml`, comparison JSON, central tar | `visual_comparison.html` | Generate visual comparison charts (FL vs CEN) |
| 8. Finalize | `bin/finalize_report.py` | `experiment_config.yaml`, central tar, configs tar | `final_report.json` | Summarize pipeline results |
| 9. Visualize | `bin/visualize.py` | `experiment_config.yaml`, central tar, configs tar | `pipeline_report.html` | Generate HTML report with charts and precipitation maps |

### Data Flow

Each job is self-contained — it extracts input tar(s) to a local temp directory, does its work, and packages output into a new tar:

```
download_config.yaml ──> download_{site} ──> {site}_raw.tar.gz
                                                    │
experiment_config.yaml ──> preproc_{site} ──> {site}_sequences.tar.gz
                                                    │
experiment_config.yaml ──> snapshot_{site} ──> {site}_snapshot.tar.gz
                                                    │
                          central_snapshot ──> central_snapshot.tar.gz
                                                    │
                          prepare_configs  ──> fl_configs.tar.gz
                                                    │
                        model_compare_run  ──> model_comparison.json   (staged out)
                                                    │
                          visual_compare   ──> visual_comparison.html  (staged out)

                          finalize_report  ──> final_report.json       (staged out)
                          visualize ─────────> pipeline_report.html    (staged out)
```

## Configuration

Two YAML config files drive the workflow:

- **`download_config.yaml`** — S3 source, product, date range, station list with lat/lon, split settings
- **`experiment_config.yaml`** — sites, splits, time windows, scan range, freeze thresholds, FL params, preprocessor settings, model comparison settings

No placeholder tokens or path substitution. Configs are used as-is by every job.

### Key config sections

**experiment_config.yaml:**
- `sites` — list of radar site identifiers (e.g. KBOX, KBYX, ...)
- `splits` — data splits (train, test)
- `windows` — time window definitions with enabled flags (1mo, 3mo, ...)
- `scan` — date range (start/end)
- `freeze` — data completeness thresholds per split
- `fl` — federated learning parameters (server, rounds, concurrency)
- `preprocessor` — sequence length, gap tolerance, normalization tags
- `model_compare` — MCT endpoint URL (optional), target accuracy threshold

**download_config.yaml:**
- `stations` — radar stations with latitude/longitude/region
- `start_date` / `end_date` — download date range
- `split` — train/test split ratio and seed
- S3 source config (product, retries, clip degrees)

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate the DAG

```bash
python workflow_generator.py \
    --download-config download_config.yaml \
    --experiment-config experiment_config.yaml \
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
```

## Testing

A minimal test harness runs all 9 stages locally without Pegasus, using reduced data (1 site, 1 day, hourly steps = ~24 S3 downloads).

```bash
# Full test (downloads from S3 + runs all stages)
bash test/run_test.sh

# Re-run without re-downloading (skips S3 if tar exists)
bash test/run_test.sh --skip-download
```

Test config differences from production:

| Setting | Production | Test |
|---------|-----------|------|
| Sites | 7 | 1 (KBOX) |
| Date range | 1 month | 1 day |
| Time step | 2 min (720/day) | 60 min (24/day) |
| Sequence length | 16 | 4 |
| Total downloads | ~140,000 | ~24 |
| Freeze tolerance | 0.25/0.85 | 0.99 (relaxed) |

Test outputs are written to `test/workdir/`. View the HTML reports:

```bash
open test/workdir/pipeline_report.html
open test/workdir/visual_comparison.html
```

## Container

The workflow uses a Singularity container pulled from `docker://kthare10/sprite-fl:latest`. No bind mounts are required — all jobs operate within their Pegasus-managed working directory.

To build locally:

```bash
docker build -t kthare10/sprite-fl:latest -f Docker/Dockerfile .
docker push kthare10/sprite-fl:latest
```

## Architecture

- **Tar-based data flow**: Each job produces a tar.gz archive as its output File. Pegasus stages these between jobs automatically.
- **Explicit dependencies**: `infer_dependencies=True` with `add_inputs()`/`add_outputs()` on Pegasus File objects wires the DAG.
- **Config-driven fan-out**: `workflow_generator.py` reads `experiment_config.yaml` at generation time to create N parallel per-site branches (download -> preproc -> snapshot).
- **Fan-in aggregation**: `central_snapshot` accepts all per-site snapshot tars and merges them with `__{SITE}` suffixed sequence directories.
- **Model comparison**: `model_compare_run` compares FL vs centralized results using training logs/checkpoints. Optionally submits to a Model Compare Tool (MCT) web service.
- **Relative paths only**: Training configs generated by `prepare_configs.py` use relative data paths, making them portable across execution environments.
- **No shared state**: No SQLite database, no filesystem symlinks, no absolute paths. Each job is fully sandboxed.

## Output

Four staged-out artifacts are produced:

### `model_comparison.json`

Machine-readable comparison of FL vs centralized training:

```json
{
  "stage": "model_compare_run",
  "status": "success",
  "comparisons": [
    {
      "window": "1mo",
      "span": "2022-02",
      "fl": { "loss": 0.234, "accuracy": 0.891 },
      "cen": { "loss": 0.198, "accuracy": 0.903 },
      "delta": { "loss": 0.036, "accuracy": -0.012 },
      "per_site_fl": { "KBOX": {...}, ... },
      "convergence": { "fl_rounds_to_target": 25, "cen_epochs_to_target": 18 }
    }
  ],
  "summary": { "mean_fl_accuracy": 0.891, "mean_cen_accuracy": 0.903, ... }
}
```

### `visual_comparison.html`

Self-contained HTML report comparing FL and centralized approaches:

- **Accuracy/loss bar charts** — FL vs CEN per window/span
- **Convergence curves** — training loss and accuracy over rounds/epochs
- **Per-site FL performance** — client-level accuracy breakdown
- **Delta heatmap** — metric differences with significance indicators
- **Summary statistics table** with t-test p-values

### `final_report.json`

Machine-readable pipeline summary with sequence counts, split distributions, and training config counts.

### `pipeline_report.html`

Self-contained HTML report with embedded visualizations:

- **DAG diagram** — visual representation of the workflow structure
- **Sequences per site** — bar chart showing data distribution across radar stations
- **Data split heatmap** — train/test sequence counts per site
- **Precipitation sample maps** — spatial heatmaps from sample .nc files for each site

All HTML reports use base64-embedded PNG images and require no external dependencies to view.

### Intermediate artifacts

The `fl_configs.tar.gz` intermediate tar contains per-run config files (`config.yaml`, `clients.map.json`, `flwr.override.toml`) ready for use by downstream FL training infrastructure.
