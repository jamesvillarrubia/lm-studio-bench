# lmstudio-bench

`lmstudio-bench` is a CLI for tuning `llama.cpp` inference performance on local GGUF models (especially LM Studio workflows).

It benchmarks prompt throughput and token-generation throughput separately, stores all raw runs, and regenerates comparison reports from saved data.

## Current status

- Stable for practical tuning workflows (`setup`, `scan`, `compare`, `report`)
- Bundled as a single executable entry at `dist/cli.js` via `tsup`
- Tested with Node 22+ on Apple Silicon and Homebrew `llama.cpp`

## Prerequisites

- Node.js `>=22`
- `pnpm`
- `llama.cpp` installed (`llama-bench`, `llama-cli` on your PATH)
- Local GGUF model(s), for example under `~/.lmstudio/models`

Install `llama.cpp` on macOS:

```bash
brew install llama.cpp
```

## Installation

### Option 1: Run from source (recommended while iterating)

```bash
git clone https://github.com/<your-org>/lmstudio-bench.git
cd lmstudio-bench
pnpm install
pnpm build
```

Run commands with:

```bash
pnpm exec lmstudio-bench --help
```

### Option 2: Package publish workflow (maintainers)

```bash
pnpm release:check
pnpm pack
```

This validates tests/types/build and creates a publishable tarball.

## Getting started

1. Discover tools/hardware/models:

```bash
pnpm exec lmstudio-bench setup
```

2. Run a sensitivity scan for one model:

```bash
pnpm exec lmstudio-bench scan --model "<model-name>"
```

3. Compare two models on baseline config:

```bash
pnpm exec lmstudio-bench compare -m "<model-a>" -m "<model-b>"
```

4. Regenerate CSVs from existing runs:

```bash
pnpm exec lmstudio-bench report -i results
```

Detailed walkthrough: `docs/getting-started.md`.

## Output files

- `results/<model>/runs.jsonl`: one record per benchmark attempt
- `results/<model>/summary.csv`: per-config medians and success/failure counts
- `results/comparison.csv`: cross-model best rows

## Bundling and local development

- Build bundle: `pnpm bundle`
- Typecheck: `pnpm typecheck`
- Test: `pnpm test`
- End-to-end release check: `pnpm release:check`

## Notes for sharing on GitHub

- Replace placeholder metadata in `package.json`:
  - `repository.url`
  - `bugs.url`
  - `homepage`
- Keep `README.md`, `LICENSE`, and `dist` included for packaging (`files` field already configured).
# lm-studio-bench
