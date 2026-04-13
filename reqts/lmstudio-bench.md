# Feature: lmstudio-bench

## Problem

Running local LLMs via LM Studio on Apple Silicon requires tuning many llama.cpp parameters (context length, batch size, GPU layers, KV cache type, threads, flash attention, etc.) to find optimal performance for a given model and hardware combination. There is no off-the-shelf tool that systematically sweeps these parameters, measures the right metrics, and surfaces actionable recommendations. Most people benchmark incorrectly by collapsing distinct performance regimes into a single number.

## Goals

- [ ] Provide a CLI tool that benchmarks llama.cpp inference across arbitrary parameter configurations
- [ ] Measure three distinct regimes separately: prompt processing speed, token generation speed, and time to first token
- [ ] Support layered benchmarking: quick sensitivity scan → focused grid sweep
- [ ] Compare models head-to-head on identical configurations
- [ ] Generate reports at three tiers: terminal tables, CSV/JSON data, self-contained HTML dashboard
- [ ] Bootstrap llama.cpp installation and model discovery automatically
- [ ] Design for open-source distribution via npm/npx

## Non-Goals

- Quality/accuracy benchmarking (eval harness, perplexity measurement)
- Automated model downloading or management
- Integration with LM Studio's internal API or plugin system
- Supporting non-llama.cpp backends (MLX, vLLM, etc.) in v1
- Real-time monitoring or continuous benchmarking daemon

## Technology

- **Language:** TypeScript (strict mode, ES2022+)
- **Runtime:** Bun primary, Node.js 22 fallback
- **Bundler:** tsup
- **Validation:** Zod
- **CLI framework:** Commander
- **Benchmark engine:** llama.cpp CLI tools (llama-bench, llama-cli) via subprocess

## Architecture

### CLI Commands

| Command | Purpose |
|---------|---------|
| `lmstudio-bench setup` | Install llama.cpp (Homebrew), detect hardware, discover GGUF models, generate baseline config |
| `lmstudio-bench scan -m <model>` | Quick sensitivity analysis — vary one param at a time from baseline |
| `lmstudio-bench sweep -c config.yaml` | Full or focused grid search across parameter space |
| `lmstudio-bench compare -m a.gguf -m b.gguf` | Head-to-head model comparison on fixed or best-known configs |
| `lmstudio-bench report -i results/` | Regenerate reports from existing JSONL result data |

### Core Components

```
src/
├── cli/           CLI entry point and command handlers
├── config/        Zod schemas, YAML loading, hardware-aware defaults
├── runner/        Subprocess spawning (llama-bench, llama-cli), output parsing
├── sweep/         Sensitivity scan engine, grid search engine, run scheduler
├── reporter/      Terminal tables, CSV/JSON export, HTML dashboard generation
├── bootstrap/     Homebrew install, hardware detection, model discovery
├── types/         Shared interfaces
└── utils/         Bun/Node runtime abstraction, path resolution
```

### Bun/Node Abstraction

A single `runtime.ts` module exports `spawn()` (wraps `Bun.spawn` or `child_process.spawn`) and `isBun` (boolean). All other code uses this abstraction — the Bun/Node distinction is contained to one file.

## Parameter Space

### Performance Parameters

| Parameter | llama.cpp flag | Type | Scan range |
|-----------|---------------|------|------------|
| Context length | `-c` | int | 2048, 4096, 8192, 16384, 32768, 65536 |
| Batch size | `-b` | int | 128, 256, 512, 1024 |
| Micro-batch size | `-ub` | int | 32, 64, 128, 256 |
| GPU layers | `-ngl` | int | 0, 16, 24, 28, 32, 99 |
| Threads (generation) | `-t` | int | 4, 6, 8, 10, 12 |
| Threads (batch/prompt) | `-tb` | int | 4, 6, 8, 10, 12 |

### Memory/Cache Parameters

| Parameter | llama.cpp flag | Type | Scan range |
|-----------|---------------|------|------------|
| KV cache type (key) | `-ctk` | enum | f16, q8_0, q4_0 |
| KV cache type (value) | `-ctv` | enum | f16, q8_0, q4_0 |
| Flash attention | `-fa` | bool | on, off |
| Memory map | `--mmap` / `--no-mmap` | bool | on, off |
| Memory lock | `--mlock` | bool | on, off |

### Workload Parameters (configurable, not swept)

| Parameter | Flag | Purpose |
|-----------|------|---------|
| Prompt tokens | `-pp` | Prompt processing benchmark size |
| Token generation count | `-tg` | Generation benchmark length |
| Repetitions | `-r` | Runs per config for statistical stability |

## Configuration Format

### sweep-config.yaml

```yaml
models:
  - path: ~/.lmstudio/models/lmstudio-community/gemma-4-26b.gguf
    name: gemma-4-26b
  - path: ~/.lmstudio/models/mlx-community/qwen3.5-35b-a3b.gguf
    name: qwen3.5-35b

baseline:
  n_ctx: 4096
  n_batch: 512
  n_ubatch: 128
  n_gpu_layers: 99
  threads: 8
  threads_batch: 8
  kv_type_key: f16
  kv_type_value: f16
  flash_attention: true
  mmap: true
  mlock: false

scan:
  n_ctx: [2048, 4096, 8192, 16384, 32768]
  n_batch: [128, 256, 512, 1024]
  n_gpu_layers: [16, 24, 28, 32, 99]
  threads: [4, 6, 8, 10, 12]
  kv_type_key: [f16, q8_0, q4_0]
  kv_type_value: [f16, q8_0, q4_0]
  flash_attention: [true, false]

sweep:
  params:
    n_ctx: [4096, 8192, 16384]
    n_batch: [256, 512]
    kv_type_key: [f16, q8_0]
    kv_type_value: [f16, q8_0]

workload:
  prompt_tokens: [128, 512, 2048]
  generation_tokens: [128, 512]
  repetitions: 3

prompts:
  synthetic:
    short: "Explain TCP congestion control."
    medium: # ~500 token prompt, generated during setup from bundled prompt files
    long: # ~2000 token prompt, generated during setup from bundled prompt files
  real_world:
    coding: "Write a TypeScript function that..."
    summarize: "Summarize the following article..."
```

### .lmstudio-bench.json (local state, gitignored)

```json
{
  "llama_bench_path": "/opt/homebrew/bin/llama-bench",
  "llama_cli_path": "/opt/homebrew/bin/llama-cli",
  "hardware": {
    "chip": "Apple M2 Max",
    "cores": 12,
    "p_cores": 8,
    "e_cores": 4,
    "memory_gb": 64,
    "metal": true
  },
  "model_dirs": ["~/.lmstudio/models"],
  "models": [
    {
      "name": "gemma-4-26b-a4b",
      "path": "~/.lmstudio/models/lmstudio-community/...",
      "size_gb": 17.99
    }
  ]
}
```

## Sweep Strategy

### Scan (sensitivity analysis)

Takes the baseline config, varies ONE parameter at a time through its range, keeps everything else at baseline. Produces a sensitivity chart showing which params have the biggest impact.

**Run count:** Sum of all scan range lengths. For the default ranges above: 5 + 4 + 4 + 5 + 5 + 3 + 2 = 28 configs × repetitions.

### Sweep (focused grid)

Takes the `sweep.params` grid and runs every combination. Intended to be filled in after reviewing scan results, focusing on the parameters that actually matter.

**Run count:** Product of all sweep range lengths. For the example: 3 × 2 × 2 = 12 configs × repetitions.

## Metrics

### Primary Metrics (from llama-bench)

| Metric | Measures | Unit |
|--------|----------|------|
| Prompt processing speed (pp) | Prompt ingestion throughput | tokens/sec |
| Token generation speed (tg) | Output generation throughput | tokens/sec |
| Time to first token (TTFT) | Latency before output begins | ms |

### Secondary Metrics

| Metric | Source | Unit |
|--------|--------|------|
| Peak memory usage | llama-bench output + ps sampling | MB |
| Model load time | Timed subprocess start → ready | ms |
| Total wall time | Per-run elapsed | sec |

### Statistical Treatment

- Each config runs N times (default 3, configurable)
- Store all individual runs
- Report: median, p5, p95, standard deviation
- Median is the primary comparison metric
- Flag runs with >10% variance as "noisy"

### Real-World Prompt Metrics (from llama-cli)

Measured separately from synthetic benchmarks:
- TTFT, generation speed (tokens/sec), total completion time, output token count

## Data Storage

### Result Records

Each run produces a JSONL record:

```json
{
  "id": "run-2026-04-12T16-30-00-001",
  "timestamp": "2026-04-12T16:30:00.000Z",
  "model": "gemma-4-26b",
  "config": {
    "n_ctx": 8192,
    "n_batch": 512,
    "n_gpu_layers": 99,
    "threads": 8,
    "kv_type_key": "f16",
    "kv_type_value": "f16",
    "flash_attention": true
  },
  "workload": { "pp": 512, "tg": 128 },
  "metrics": {
    "pp_tokens_per_sec": 1245.3,
    "tg_tokens_per_sec": 42.7,
    "ttft_ms": 412,
    "peak_memory_mb": 19200,
    "load_time_ms": 3200,
    "wall_time_sec": 8.4
  },
  "run_index": 1,
  "repetitions": 3
}
```

### File Layout

- `results/<model-name>/runs.jsonl` — raw run data, one JSON object per line
- `results/<model-name>/summary.csv` — one row per config, median metrics
- `results/comparison.csv` — cross-model comparison

## Reporting

### Tier 1: Terminal

- Live progress bar during sweep/scan with current config and metrics
- Summary table after completion with all configs ranked
- "Best config" callouts for generation speed, prompt processing, and balanced

### Tier 2: CSV/JSON

- JSONL for raw data (machine-readable, appendable)
- CSV summaries for spreadsheet analysis

### Tier 3: HTML Dashboard

Self-contained single HTML file with inline CSS/JS and Chart.js:

1. **Sensitivity charts** — bar charts showing per-parameter impact on pp/tg/TTFT
2. **Heatmaps** — 2D grids of metric values across two parameter axes
3. **Model comparison** — grouped bar charts, side-by-side per model
4. **Pareto frontier** — scatter plot of tg speed vs. context length, highlighting non-dominated configs
5. **Config recommendations** — top 3 configs with rationale:
   - "Best throughput" (highest tg)
   - "Best for long context" (highest tg with n_ctx >= 16384)
   - "Best balanced" (Pareto-optimal across pp, tg, and memory)

Regenerable from JSONL data via `lmstudio-bench report`.

## Bootstrap Flow

### `lmstudio-bench setup`

1. **Check for llama.cpp** — look for `llama-bench` and `llama-cli` on PATH
2. **Install if missing** — `brew install llama.cpp`, fall back to build-from-source instructions
3. **Detect hardware** — `system_profiler SPHardwareDataType` → chip, cores, memory, Metal support
4. **Discover models** — scan `~/.lmstudio/models/` and LM Studio app data directories for `.gguf` files
5. **Interactive model selection** — user picks which models to benchmark
6. **Generate config** — hardware-aware `sweep-config.yaml` with sensible defaults and scan ranges
7. **Write local state** — `.lmstudio-bench.json` with paths, hardware info, model list

## Project Structure

```
lmstudio-bench/
├── src/
│   ├── cli/
│   │   ├── index.ts
│   │   ├── setup.ts
│   │   ├── scan.ts
│   │   ├── sweep.ts
│   │   ├── compare.ts
│   │   └── report.ts
│   ├── config/
│   │   ├── schema.ts
│   │   ├── loader.ts
│   │   └── defaults.ts
│   ├── runner/
│   │   ├── llama-bench.ts
│   │   ├── llama-cli.ts
│   │   ├── parser.ts
│   │   └── subprocess.ts
│   ├── sweep/
│   │   ├── sensitivity.ts
│   │   ├── grid.ts
│   │   └── scheduler.ts
│   ├── reporter/
│   │   ├── terminal.ts
│   │   ├── csv.ts
│   │   ├── html.ts
│   │   ├── charts.ts
│   │   └── recommend.ts
│   ├── bootstrap/
│   │   ├── install.ts
│   │   ├── hardware.ts
│   │   └── models.ts
│   ├── types/
│   │   └── index.ts
│   └── utils/
│       ├── runtime.ts
│       └── paths.ts
├── templates/
│   └── report.html
├── prompts/
│   ├── synthetic/
│   └── real-world/
├── results/                  # gitignored
├── tests/
│   ├── integration/
│   ├── unit/
│   └── fixtures/
├── package.json
├── tsconfig.json
├── tsup.config.ts
├── sweep-config.example.yaml
├── .lmstudio-bench.example.json
├── .gitignore
├── LICENSE
└── README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| zod | Config validation |
| yaml | YAML parsing |
| cli-table3 | Terminal table formatting |
| chalk | Terminal colors |
| commander | CLI argument parsing |

Zero native dependencies. Pure JS/TS for cross-platform distribution.

## Distribution

- `tsup` bundles to single ESM file targeting Node 22 / Bun
- `package.json` bin field: `{ "lmstudio-bench": "./dist/cli.js" }`
- `npx lmstudio-bench` works after npm publish
- Homebrew formula possible later (wraps npm package)

## Behavior

- When `setup` has not been run, all other commands print a helpful error directing the user to run setup first
- When a sweep/scan is interrupted (Ctrl+C), results collected so far are saved — no data is lost
- When a single benchmark run fails (e.g., OOM on large context), the error is logged and the sweep continues with remaining configs
- When `--estimate-only` is passed, the tool calculates total run count and estimated wall time without executing anything
- When models are specified by name (not path), they're resolved from the discovered model list in `.lmstudio-bench.json`

## Open Questions

- [ ] Should we support resuming interrupted sweeps (skip configs that already have results)?
- [ ] Should the HTML report include a "share" feature (export to a static hosting-friendly format)?
- [ ] Should we add a `watch` command that re-runs a specific config periodically to detect thermal throttling?
