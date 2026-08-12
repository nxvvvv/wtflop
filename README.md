# wtflop

A GPU benchmarking CLI tool. "wtflop" = **W**hat **T**he **FLOP** — measures matrix-multiply throughput, tensor ops, memory bandwidth, data transfer, and inference performance on your GPU, with live monitoring and results tracked to SQLite.

Works in two modes:
- **Interactive** — run `python wtf.py` with no args and get a guided menu (pick benchmark → configure options → confirm → run).
- **Scriptable** — pass `--benchmark <name>` plus flags for CI/automation use.

## Benchmarks

| Key | What it measures |
|---|---|
| `mamf` | Matrix multiplication throughput (Maximum Achievable FLOPs) |
| `tensor` | General tensor operation performance |
| `datagen` | GPU-side data generation throughput |
| `transfer` | GPU↔CPU and GPU↔GPU transfer bandwidth |
| `membw` | GPU and system memory bandwidth |
| `inference` | Inference throughput (e.g. BERT/GPT-style models via `transformers`) |
| `compute` | General GPU computational benchmark |

## Requirements

- Python 3.9+
- A CUDA-capable GPU (GPU monitoring auto-detects CUDA; benchmarks should still run on CPU/other backends via the `utils.arch` accelerator abstraction, though CUDA is the primary target)
- Dependencies in `requirements.txt`: PyTorch/torchvision, NumPy, pandas, SciPy, Optuna, SQLAlchemy, GPUtil, psutil, nvidia-ml-py, questionary, typer, rich, transformers, h5py, seaborn, and `pywin32` on Windows.

## Install

```bash
git clone https://github.com/nxvvvv/wtflop.git
cd wtflop
pip install -r requirements.txt
```

## Usage

### Interactive mode

```bash
python wtf.py
```

Walks you through: selecting a benchmark → reviewing/customizing its options → configuring GPU monitoring and notes → confirming the assembled command before it runs.

Add `--fast-menu` to skip the GPU info banner for quicker startup:

```bash
python wtf.py --fast-menu
```

### Command-line mode

```bash
python wtf.py --benchmark mamf [options...]
```

Common flags (apply to all benchmarks):

| Flag | Default | Description |
|---|---|---|
| `--benchmark` | `mamf` | Which benchmark to run (see table above) |
| `--output-file` | `results/logs/<benchmark>_<timestamp>.txt` | Where to save output (relative paths go under `results/logs/`) |
| `--notes` | *(empty)* | Free-text notes written into the output file header |
| `--verbose` / `--no-verbose` | `--verbose` | Also print to stdout in addition to the output file |
| `--monitor-interval` | `1.0` | Seconds between GPU monitoring samples |
| `--monitor-db` | `gpu_monitoring.db` | SQLite file for GPU monitoring data |
| `--skip-monitoring` | off | Disable GPU monitoring for this run |

Each benchmark module also exposes its own additional options (visible via the interactive menu's options table, or by inspecting `benchmarks/<name>.py`).

While a benchmark is running:
- Press **`s`** to request a graceful stop after the current trial finishes.
- Press **`s`** again (or **Ctrl+C** twice) to force an immediate stop.

## Output & results

- `results/logs/` — per-run text logs (console output mirrored to file)
- `results/db/` — SQLite databases: one per benchmark type, plus GPU monitoring data
- `results/reports/` — generated reports
- `results/benchmark_summary.json` — rolling summary of performance results (throughput, bandwidth, or GFLOPs) across runs, keyed by benchmark and config

These directories are created automatically on first run.

## Project structure

```
wtflop/
├── wtf.py              # CLI entrypoint (interactive + argparse modes)
├── benchmarks/          # Individual benchmark implementations (mamf, tensor_ops, ...)
├── utils/                # Shared helpers
│   ├── arch.py           # GPU/accelerator architecture detection
│   ├── tee.py             # Dual console+file output
│   ├── gpu_monitor.py      # Background GPU resource monitoring
│   ├── db_utils.py         # SQLite setup and summary updates
│   ├── shared_state.py      # Cross-thread termination flag
│   └── utilities.py         # Misc helpers (header printing, signal handling)
├── requirements.txt
└── LICENSE               # AGPL-3.0
```

## Why this is a genuinely useful project

- **It answers a real question people can't easily answer otherwise.** "What FLOPs/bandwidth am I actually getting out of this GPU, on this box, right now?" — as opposed to trusting spec-sheet numbers or one-off `torch.cuda` snippets people paste from forums. Having mamf, tensor ops, memory bandwidth, transfer, data generation, inference, and general compute all in one place means you get a fairly complete picture of a GPU's real-world behavior instead of just one narrow slice of it.
- **It's built for repeatable, comparable measurement, not a single throwaway run.** Every run is logged, timestamped, and written into SQLite with a rolling summary — so you can benchmark before/after a driver update, a new CUDA version, a different batch size, or swapping hardware, and actually compare results instead of re-deriving them from memory or scattered log files.
- **It bridges "quick check" and "serious benchmarking" use cases.** The interactive menu is for someone who just wants to poke at their GPU and see numbers; the CLI/argparse mode is for scripting into CI, cron jobs, or a fleet of machines. A lot of benchmarking scripts only support one of these, which limits who ends up using them.
- **Resource monitoring alongside the benchmark, not just the end number.** Sampling GPU utilization/memory over the course of a run (not just reporting a final throughput figure) is what actually lets you tell "this is compute-bound" from "this is memory-bound" or "this is thermal-throttling," which is the more useful diagnostic question in practice.
- **Low friction to actually use.** Sensible defaults (auto-named output files, results auto-organized into logs/db/reports), a graceful stop-mid-run option instead of forcing a hard kill, and being runnable with zero flags — these matter more than they sound for whether a tool actually gets used repeatedly versus written once and abandoned.
- **It's a practical fit for the kind of work it'd be used for**: comparing GPUs before a purchase, sanity-checking a new machine or cloud instance, validating that a driver/library upgrade didn't regress performance, or building intuition about where a specific workload's bottleneck actually is.

## License

[AGPL-3.0](LICENSE)
