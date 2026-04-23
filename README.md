# lmstudio-bench

A CLI for finding the fastest `llama.cpp` inference settings for your local GGUF models. It runs structured benchmarks across all the knobs that matter — threads, KV cache types, GPU layers, batch sizes, context length, Flash Attention, and more — then tells you what actually moves the needle on your hardware.

Designed for laptop use (especially Apple Silicon) with multiple models over time.

**License:** [MIT](LICENSE). **Citation (academic / reports):** see [CITATION.md](CITATION.md).

---

## How it works

All runs are stored in `~/.lmstudio-bench/results/<model>/runs.jsonl`. Each run is content-addressed by its config + hardware fingerprint, so:

- Runs are cached and never re-executed unless you ask.
- Switching models just starts a new result file — your other models' data is untouched.
- Changing hardware (e.g. adding RAM) invalidates the cache cleanly.
- You can re-run `analyze` and `visualize` at any time without re-benchmarking.

---

## Prerequisites

- **Node.js** `>=22`
- **llama.cpp** with `llama-bench` on your PATH
- One or more local **GGUF model files**

Install `llama.cpp` on macOS:

```bash
brew install llama.cpp
```

---

## Installation

### From npm (recommended)

Install the CLI globally (any package manager works):

```bash
npm install -g lmstudio-bench
# or: pnpm add -g lmstudio-bench
```

Run without installing (downloads the package on first use):

```bash
npx lmstudio-bench --help
npx lmstudio-bench setup
```

### From GitHub (no npm registry)

The published package is a **Node bundle** (`dist/cli.js`), not a separate native binary. Because `dist/` is not committed, the repo uses a **`prepare`** script so installing from git runs `tsup` and produces `dist/` automatically.

In another project:

```bash
pnpm add https://github.com/jamesvillarrubia/lm-studio-bench.git
# optional pin: …#main or …#v0.1.0
```

Or with npm:

```bash
npm install github:jamesvillarrubia/lm-studio-bench
```

Requires **Node 22+** on the machine running the install (same as runtime). Then use `pnpm exec lmstudio-bench` / `npx lmstudio-bench` from that project, or link globally if you prefer.

### Native executable (Bun, no Node at runtime)

[Bun](https://bun.sh) can compile the CLI into a **single platform binary** (~tens of MB) that does **not** require Node on the machine where you run benchmarks. You still need Bun (or CI) to **produce** that file.

From a clone of this repo:

```bash
pnpm install   # dev deps; optional if you only use Bun for this step
pnpm run build:native
./artifacts/lmstudio-bench --help
```

Cross-compile for another OS (examples):

```bash
BUN_COMPILE_TARGET=bun-linux-x64 pnpm run build:native
BUN_COMPILE_TARGET=bun-windows-x64 pnpm run build:native   # writes artifacts/lmstudio-bench.exe
```

GitHub Actions workflow **`bun-native.yml`** builds Linux (x64, arm64), macOS (arm64, x64), and Windows (x64) and uploads each as a workflow artifact (download from the Actions run).

The executable name is **`lmstudio-bench`**. The rest of this README uses that command. From a git checkout of this repo, `pnpm install` runs `prepare` and builds `dist/`; you can still run `pnpm exec lmstudio-bench …` or `node dist/cli.js …`.

### From source (contributors)

```bash
git clone https://github.com/jamesvillarrubia/lm-studio-bench.git
cd lm-studio-bench
pnpm install
pnpm build
pnpm exec lmstudio-bench --help
```

---

## Publishing to npm (maintainers)

From a clean tree, with an [npmjs.com](https://www.npmjs.com) account and `npm login`:

```bash
npm version patch   # or minor / major
npm publish --access public
```

`prepublishOnly` runs `npm run build` and `npm test` automatically. Ensure the package name `lmstudio-bench` is available on npm (or change `"name"` in `package.json` before the first publish).

### GitHub Actions and PipeCraft

This repository is wired for [PipeCraft](https://pipecraft.thecraftlab.dev) (`pipecraft` on npm). The file `.pipecraftrc` describes branch flow (`develop` → `main`), domains for change detection, and GitHub as the CI provider. Running `pnpm workflow:generate` regenerates `.github/workflows/pipeline.yml` and the composite actions under `.github/actions/` from that config (keep edits inside the `# <--START CUSTOM JOBS-->` / `# <--END CUSTOM JOBS-->` block in `pipeline.yml`, and keep any manual adjustments to `gate.needs` / `gate.if` when you add jobs there). `pnpm workflow:validate` checks the config only.

The generated pipeline computes versions from conventional commits, can tag and promote, and creates GitHub Releases on `main`. After a release is created, PipeCraft’s `create-release` action looks for `.github/workflows/publish.yml` and dispatches it with the new tag so you can publish to npm in CI. Add a repository secret named `NPM_TOKEN` (npm automation access token with publish permission for this package). Optional: run `pnpm exec pipecraft setup-github` (or `--apply`) so branch protection and workflow permissions match PipeCraft’s expectations.

---

## Full workflow

### Step 1 — One-time setup

Run this once per machine (or after upgrading llama.cpp):

```bash
lmstudio-bench setup
```

This probes your `llama-bench` binary for supported flags, detects hardware (CPU cores, RAM, GPU), and writes `~/.lmstudio-bench/state.json`. It also discovers any GGUF models under `~/.lmstudio/models`.

---

### Step 2 — Configure your sweep

Edit `sweep-config.yaml` in this repo. It has four sections:

#### `models`
List the models you want to benchmark. Add as many as you like; each model gets its own results directory.

```yaml
models:
  - path: /Users/you/.lmstudio/models/org/model-Q4_K_M.gguf
    name: my-model-Q4_K_M
  - path: /Users/you/.lmstudio/models/org/model-Q8_0.gguf
    name: my-model-Q8_0
```

#### `baseline`
The default settings used as the starting point for every sweep variation. Set this to your best current guess — or just leave the defaults.

```yaml
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
  no_kv_offload: false
  n_cpu_moe: 0        # Only relevant for MoE models (e.g. Gemma MoE, Mixtral)
```

#### `scan`
The values to sweep for each parameter. You generally don't need to change this unless you want to add or remove levels.

#### `workload`
How many tokens each benchmark run generates. Larger values give more stable measurements but take longer.

```yaml
workload:
  prompt_tokens: [512]
  generation_tokens: [128]
  repetitions: 3          # Median of 3 runs per config — increase for noisier hardware
```

> The config file at `~/.lmstudio-bench/sweep-config.yaml` is a symlink to this file, so edits here take effect everywhere automatically.

---

### Step 3 — Broad exploration with adaptive scan

The adaptive strategy uses Latin Hypercube Sampling to cover the entire parameter space efficiently, then focuses a second pass around the top performers.

```bash
# Estimate run count first (no benchmarks run)
lmstudio-bench scan --strategy adaptive --estimate-only

# Run it (verbose shows progress per run)
lmstudio-bench scan --strategy adaptive -v
```

For a typical sweep config (~12 axes, 60–90 levels total), expect **~200–350 configs** in phase 1 plus a focused grid in phase 2. With 3 reps and a fast model, budget 2–4 hours.

To scan a specific model when you have multiple in your config:

```bash
lmstudio-bench scan -m my-model-Q4_K_M --strategy adaptive -v
```

If the run is interrupted, just re-run the same command. Cached runs are skipped automatically.

---

### Step 4 — Identify the important factors

Once you have sweep data, run the statistical analysis:

```bash
lmstudio-bench analyze -m my-model-Q4_K_M
```

This fits a regression model to your runs and outputs a ranked table of factors by their contribution to throughput variance:

```
Model fit: n=612, unique=204, r2=0.7812, adj_r2=0.7634

┌──────────────────┬────────────┬───────┬──────────────────────────┬────────┬──────────┬───────────┬──────────┐
│ factor           │ importance │ share │ 95% CI                   │ F      │ p        │ direction │ nonlinear│
├──────────────────┼────────────┼───────┼──────────────────────────┼────────┼──────────┼───────────┼──────────┤
│ threads          │ 0.18340    │ 28.4% │ [0.14210, 0.22890]       │ 84.32  │ 1.2e-14  │ positive  │ no       │
│ n_gpu_layers     │ 0.14120    │ 21.9% │ [0.10880, 0.17640]       │ 62.44  │ 4.1e-12  │ positive  │ no       │
│ kv_type_key      │ 0.09210    │ 14.3% │ [0.06330, 0.12410]       │ 38.91  │ 2.3e-9   │ mixed     │ no       │
│ flash_attention  │ 0.06480    │ 10.1% │ [0.04120, 0.09140]       │ 27.14  │ 1.1e-7   │ positive  │ no       │
│ ...              │            │       │                          │        │          │           │          │
└──────────────────┴────────────┴───────┴──────────────────────────┴────────┴──────────┴───────────┴──────────┘
```

Key columns:
- **importance** — share of throughput variance explained by this factor alone
- **share** — percentage of total explained variance
- **95% CI** — bootstrap confidence interval (wide CI = noisy estimate, run more reps)
- **direction** — does higher = faster (`positive`), lower = faster (`negative`), or mixed?
- **nonlinear** — does the factor have a non-monotonic sweet spot?

Save a JSON report for use in external tools:

```bash
lmstudio-bench analyze -m my-model-Q4_K_M --json
# Writes: ~/.lmstudio-bench/results/my-model-Q4_K_M/analysis.json
```

---

### Step 5 — Visualize

Generate an interactive HTML report with charts, recommended settings, and an interaction heatmap:

```bash
lmstudio-bench visualize -m my-model-Q4_K_M
# Opens: ~/.lmstudio-bench/results/my-model-Q4_K_M/plots/impact-report.html
```

The report includes:
- **Recommended Settings** — the best observed config across all runs, plus per-factor recommendations derived from the statistical model
- **Factor Importance chart** — horizontal bar chart with bootstrap confidence intervals
- **Per-axis throughput plots** — how each setting affects tg and pp speed
- **Interaction heatmap** — which pair of factors has the strongest combined effect

Open the HTML file in any browser; it has no external dependencies.

---

### Step 6 — Confirmatory scan

After the adaptive sweep identifies the best config and the analysis ranks the important factors, run a clean one-at-a-time scan pinned to the best observed configuration. This isolates each axis's true effect at the optimum and confirms there's no interaction hiding a better setting.

```bash
lmstudio-bench scan --strategy confirmatory -v
```

This automatically reads your `runs.jsonl`, picks the config with the highest median tg throughput, and sweeps every scan axis around it one at a time. Most runs will be cache hits from the adaptive sweep; only the combinations not previously tested will actually execute.

To confirm against the 2nd-best config instead (useful when the top config looks like a noise spike):

```bash
lmstudio-bench scan --strategy confirmatory --confirmatory-top-n 2
```

---

### Step 7 — Lock in your settings

After the confirmatory scan, re-run `analyze` and `visualize` to see the sharpened picture:

```bash
lmstudio-bench analyze -m my-model-Q4_K_M
lmstudio-bench visualize -m my-model-Q4_K_M
```

Update the `baseline` in `sweep-config.yaml` with the confirmed best values. This becomes the starting point for future scans — including when you add a new model.

---

## Repeating the workflow for new models

Adding a second or third model over time requires minimal additional work:

1. Add the model to the `models:` list in `sweep-config.yaml`.
2. Run the adaptive scan for that model:
   ```bash
   lmstudio-bench scan -m new-model-name --strategy adaptive -v
   ```
3. Run `analyze` and `visualize` for the new model.
4. Optionally run a confirmatory scan.

The baseline you've already tuned on your hardware is a reasonable starting point for any model of similar size. Models with different architectures (MoE vs dense, different layer counts) may show different optimal settings, so it's worth running the full flow once per model class.

### MoE models

For models like Gemma MoE or Mixtral, the `n_cpu_moe` parameter controls how many expert layers run on CPU. Make sure it's in your scan:

```yaml
scan:
  n_cpu_moe:
    - 0
    - 2
    - 4
    - 6
    - 8
```

Set it to `0` in the baseline if you're fully GPU-offloading.

---

## Comparing models directly

To benchmark two models head-to-head with a fixed config:

```bash
lmstudio-bench compare -m model-a -m model-b
```

This runs both models on the baseline config and outputs a side-by-side throughput comparison. Useful for quantization comparisons (Q4 vs Q8) or before/after a model update.

---

## Commands reference

| Command | What it does |
|---|---|
| `setup` | One-time: probe llama-bench capabilities, detect hardware and models |
| `list-models` | Show models discovered by setup |
| `scan` | Run benchmarks: `--strategy oat`, `adaptive`, or `confirmatory` |
| `analyze` | Statistical factor-importance analysis (regression + bootstrap) |
| `visualize` | Generate interactive HTML report with charts and recommendations |
| `compare` | Head-to-head baseline comparison across two or more models |
| `report` | Regenerate CSV summaries from existing `runs.jsonl` files |

Each command accepts `--help` for full options.

---

## Data files

| Path | Contents |
|---|---|
| `~/.lmstudio-bench/state.json` | Hardware profile, llama-bench path, discovered models |
| `~/.lmstudio-bench/sweep-config.yaml` | Symlink to your project's `sweep-config.yaml` |
| `~/.lmstudio-bench/results/<model>/runs.jsonl` | Every benchmark run ever executed for that model |
| `~/.lmstudio-bench/results/<model>/summary.csv` | Per-config medians (regenerated by `report`) |
| `~/.lmstudio-bench/results/<model>/analysis.json` | Statistical factor analysis output (`analyze --json`) |
| `~/.lmstudio-bench/results/<model>/plots/impact-report.html` | Interactive visualization (`visualize`) |

---

## Cache behavior

- **Same config + hardware + run index**: cache hit, no re-execution.
- **Changed hardware** (e.g. different machine, RAM change): new runs, old data preserved.
- **Changed `repetitions`**: individual rep results are reused; only new reps execute.
- **Force re-run**: `--rerun` ignores all cache. `--retry-failed` re-runs only failures.

---

## Development

```bash
pnpm build        # Compile TypeScript → dist/cli.js
pnpm test         # Run test suite (Vitest)
pnpm typecheck    # Type-check without emitting
pnpm release:check  # Full pre-release gate: typecheck + test + build
```
