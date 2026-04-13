import path from "path";
import chalk from "chalk";
import type { Command } from "commander";
import { fileExists, readJson, readYaml } from "../config/loader.js";
import {
  LocalStateSchema,
  SweepConfigSchema,
  RunRecordSchema,
} from "../config/schema.js";
import type { LocalState, RunRecord, SweepConfig } from "../config/schema.js";
import {
  applyModelOverrides,
  filterToAppliedBenchmarkConfig,
  filterScanForCapabilities,
  findUnsupportedBaselineKeys,
  findUnsupportedScanKeys,
} from "../config/resolution.js";
import { probeLlamaBenchCapabilities } from "../runner/llama-bench-capabilities.js";
import type { BenchCapabilities } from "../runner/llama-bench-capabilities.js";
import { inferLlamaBenchBinaryFingerprint } from "../runner/bench-binary-fingerprint.js";
import { createBenchInput } from "../runner/llama-bench.js";
import type { BenchRunner } from "../runner/llama-bench.js";
import { computeRunKey } from "../cache/run-key.js";
import { expandSensitivityScan } from "../sweep/sensitivity.js";
import {
  appendRunRecord,
  loadRunRecords,
  findLatestRunRecordByRunKey,
  summarizeRecords,
} from "../reporter/results.js";
import { printScanSummary } from "../reporter/terminal.js";
import type { LoggerLike } from "../reporter/terminal.js";

function resolveModelPath(
  state: LocalState,
  modelRef: string | undefined,
  config: SweepConfig,
): { modelName: string; modelPath: string } {
  if (!modelRef) {
    const first = config.models[0];
    if (!first) {
      throw new Error("No models found in sweep config");
    }
    return { modelName: first.name, modelPath: first.path };
  }
  if (modelRef.includes("/") || modelRef.endsWith(".gguf")) {
    return {
      modelName: path.basename(modelRef, path.extname(modelRef)),
      modelPath: modelRef,
    };
  }
  const fromConfig = config.models.find((item) => item.name === modelRef);
  if (fromConfig) {
    return { modelName: fromConfig.name, modelPath: fromConfig.path };
  }
  const fromState = state.models.find((item) => item.name === modelRef);
  if (fromState) {
    return { modelName: fromState.name, modelPath: fromState.path };
  }
  throw new Error(`Model '${modelRef}' could not be resolved`);
}

function sanitizeFilePart(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "-");
}

function tuningSnapshotToRecord(
  source: Record<string, unknown>,
): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(source)) {
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      out[key] = value;
    }
  }
  return out;
}

function buildFailedRecord(
  base: Record<string, unknown>,
  message: string,
): RunRecord {
  return RunRecordSchema.parse({
    ...base,
    metrics: {
      pp_tokens_per_sec: null,
      tg_tokens_per_sec: null,
      ttft_ms: null,
      peak_memory_mb: null,
      load_time_ms: null,
      wall_time_sec: null,
    },
    status: "failed",
    error: { message },
  });
}

function buildSuccessRecord(
  base: Record<string, unknown>,
  metrics: {
    ppTokensPerSec: number;
    tgTokensPerSec: number;
    ttftMs: number | null;
    wallTimeSec?: number;
  },
): RunRecord {
  return RunRecordSchema.parse({
    ...base,
    metrics: {
      pp_tokens_per_sec: metrics.ppTokensPerSec,
      tg_tokens_per_sec: metrics.tgTokensPerSec,
      ttft_ms: metrics.ttftMs,
      wall_time_sec: metrics.wallTimeSec ?? null,
      peak_memory_mb: null,
      load_time_ms: null,
    },
    status: "success",
    error: null,
  });
}

function buildCacheHitRecord(
  base: Record<string, unknown>,
  reused: RunRecord,
): RunRecord {
  return RunRecordSchema.parse({
    ...base,
    metrics: reused.metrics,
    status: "success",
    error: null,
    cache: {
      hit: true,
      reused_run_id: reused.id,
    },
  });
}

export interface ScanDeps {
  cwd?: () => string;
  now?: () => Date;
  logger?: LoggerLike;
  benchRunner: BenchRunner;
  benchCapabilities?: BenchCapabilities;
}

export function registerScanCommand(
  command: Command,
  deps: ScanDeps,
): void {
  const cwd = deps.cwd ?? (() => process.cwd());
  const now = deps.now ?? (() => new Date());
  const logger = deps.logger ?? console;

  command
    .command("scan")
    .description("Run one-parameter sensitivity scan")
    .option("-m, --model <model>", "Model name or path")
    .option("-c, --config <path>", "Config path", "sweep-config.yaml")
    .option(
      "--estimate-only",
      "Estimate run count without executing",
    )
    .option(
      "--rerun",
      "Ignore successful benchmark cache hits and execute all planned runs",
    )
    .option(
      "--retry-failed",
      "Skip successful cache hits, but rerun failures (based on latest record per run key)",
    )
    .option("-v, --verbose", "Print detailed progress for each run")
    .action(
      async (options: {
        model?: string;
        config?: string;
        estimateOnly?: boolean;
        rerun?: boolean;
        retryFailed?: boolean;
        verbose?: boolean;
      }) => {
        const root = cwd();
        const statePath = path.join(root, ".lmstudio-bench.json");
        if (!(await fileExists(statePath))) {
          throw new Error("Run `lmstudio-bench setup` first");
        }
        const configPath = path.join(
          root,
          options.config ?? "sweep-config.yaml",
        );
        const state = await readJson(statePath, LocalStateSchema);
        const capabilities =
          deps.benchCapabilities ??
          (await probeLlamaBenchCapabilities(state.llama_bench_path));
        for (const warning of capabilities.warnings) {
          logger.log(`llama-bench capabilities: ${warning}`);
        }
        const config = await readYaml(configPath, SweepConfigSchema);
        const model = resolveModelPath(state, options.model, config);
        const resolvedConfig = applyModelOverrides(config, {
          name: model.modelName,
          path: model.modelPath,
        });
        const unsupportedBaselineKeys = findUnsupportedBaselineKeys(
          resolvedConfig.baseline,
          capabilities,
        );
        if (unsupportedBaselineKeys.length > 0) {
          logger.log(
            `Ignoring unsupported baseline fields for this llama-bench build: ${unsupportedBaselineKeys.join(", ")}`,
          );
        }
        const unsupportedScanKeys = findUnsupportedScanKeys(
          resolvedConfig.scan,
          capabilities,
        );
        if (unsupportedScanKeys.length > 0) {
          logger.log(
            `Ignoring unsupported scan axes for this llama-bench build: ${unsupportedScanKeys.join(", ")}`,
          );
        }
        const appliedScan = filterScanForCapabilities(
          resolvedConfig.scan,
          capabilities,
        );

        const verbose = Boolean(options.verbose);
        if (verbose) {
          logger.log(chalk.bold("\n\u2500\u2500 Capabilities \u2500\u2500"));
          logger.log(`  Binary: ${state.llama_bench_path}`);
          logger.log(
            `  Supported keys: ${[...capabilities.supportedKeys].sort().join(", ")}`,
          );
          if (capabilities.usedFallback) {
            logger.log(
              chalk.yellow(
                "  (used fallback defaults \u2014 could not probe binary)",
              ),
            );
          }
          logger.log(chalk.bold("\n\u2500\u2500 Scan Plan \u2500\u2500"));
          const scanAxes = Object.entries(appliedScan).filter(
            ([, v]) => Array.isArray(v) && (v as unknown[]).length > 0,
          );
          for (const [axis, values] of scanAxes) {
            logger.log(
              `  ${axis}: [${(values as unknown[]).join(", ")}]`,
            );
          }
          logger.log(chalk.bold("\n\u2500\u2500 Baseline \u2500\u2500"));
          for (const [key, value] of Object.entries(
            resolvedConfig.baseline,
          )) {
            logger.log(`  ${key}: ${JSON.stringify(value)}`);
          }
        }

        const entries = expandSensitivityScan(
          resolvedConfig.baseline as Record<string, string | number | boolean>,
          appliedScan,
        );
        const runCount =
          entries.length *
          resolvedConfig.workload.prompt_tokens.length *
          resolvedConfig.workload.generation_tokens.length *
          resolvedConfig.workload.repetitions;

        if (verbose) {
          logger.log(
            chalk.bold("\n\u2500\u2500 Run Estimate \u2500\u2500"),
          );
          logger.log(
            `  ${entries.length} configs \u00D7 ${resolvedConfig.workload.prompt_tokens.length} prompt sizes \u00D7 ${resolvedConfig.workload.generation_tokens.length} gen sizes \u00D7 ${resolvedConfig.workload.repetitions} reps = ${chalk.cyan(String(runCount))} total runs`,
          );
          logger.log(
            `  Cache mode: ${options.rerun ? "rerun all (ignoring cache)" : options.retryFailed ? "retry failed only" : "skip cached successes"}`,
          );
          logger.log("");
        }

        if (options.estimateOnly) {
          logger.log(`Estimated runs: ${runCount}`);
          return;
        }

        const runsFilePath = path.join(
          root,
          "results",
          sanitizeFilePart(model.modelName),
          "runs.jsonl",
        );
        const historicalRecords = await loadRunRecords(runsFilePath);
        const runnerIdentity =
          (await deps.benchRunner.getRunnerIdentity?.({
            llamaBenchPath: state.llama_bench_path,
          })) ??
          (await inferLlamaBenchBinaryFingerprint(
            state.llama_bench_path,
          ));
        const records: RunRecord[] = [];
        let runIndex = 0;
        let hadFailures = false;
        let cacheHits = 0;
        let executed = 0;
        let failed = 0;
        const scanStartTime = Date.now();

        for (const entry of entries) {
          for (const promptTokens of resolvedConfig.workload
            .prompt_tokens) {
            for (const generationTokens of resolvedConfig.workload
              .generation_tokens) {
              for (
                let repetition = 1;
                repetition <= resolvedConfig.workload.repetitions;
                repetition += 1
              ) {
                runIndex += 1;
                const appliedRow =
                  filterToAppliedBenchmarkConfig(
                    entry.config,
                    capabilities,
                  );
                const runKey = computeRunKey({
                  benchCommand: "scan",
                  modelPath: model.modelPath,
                  appliedConfig: appliedRow,
                  workload: {
                    pp: promptTokens,
                    tg: generationTokens,
                  },
                  repetition: {
                    runIndex: repetition,
                    repetitions:
                      resolvedConfig.workload.repetitions,
                  },
                  hardware: state.hardware as unknown as Record<
                    string,
                    unknown
                  >,
                  runnerIdentity,
                });
                const requestedRecord = tuningSnapshotToRecord(
                  entry.config,
                );
                const appliedRecord =
                  tuningSnapshotToRecord(appliedRow);
                const configSummary = Object.entries(appliedRecord)
                  .map(([k, v]) => `${k}=${v}`)
                  .join(" ");
                const progressPrefix = `[${runIndex}/${runCount}]`;
                const baseRecord = {
                  id: `run-${now().toISOString().replace(/[:.]/g, "-")}-${runIndex}`,
                  timestamp: now().toISOString(),
                  model: model.modelName,
                  run_key: runKey,
                  bench_command: "scan" as const,
                  runner_identity: runnerIdentity,
                  hardware: state.hardware,
                  requested_config: requestedRecord,
                  applied_config: appliedRecord,
                  config: appliedRecord,
                  capability_meta: {
                    supported_keys: [
                      ...capabilities.supportedKeys,
                    ].sort(),
                    used_fallback: capabilities.usedFallback,
                    warnings:
                      capabilities.warnings.length > 0
                        ? [...capabilities.warnings]
                        : undefined,
                  },
                  workload: {
                    pp: promptTokens,
                    tg: generationTokens,
                  },
                  run_index: repetition,
                  repetitions:
                    resolvedConfig.workload.repetitions,
                };

                const latest = findLatestRunRecordByRunKey(
                  [...historicalRecords, ...records],
                  runKey,
                );
                const rerun = Boolean(options.rerun);
                const retryFailed = Boolean(options.retryFailed);
                const canReuseLatestSuccess =
                  !rerun && latest?.status === "success";

                if (canReuseLatestSuccess) {
                  cacheHits += 1;
                  const record = buildCacheHitRecord(
                    baseRecord,
                    latest,
                  );
                  records.push(record);
                  if (verbose) {
                    logger.log(
                      `${chalk.dim(progressPrefix)} ${chalk.yellow("CACHED")} pp=${promptTokens} tg=${generationTokens} | ${configSummary}`,
                    );
                  }
                  continue;
                }

                if (verbose) {
                  logger.log(
                    `${chalk.cyan(progressPrefix)} ${chalk.bold("RUNNING")} pp=${promptTokens} tg=${generationTokens} rep=${repetition} | ${configSummary}`,
                  );
                }

                const runStart = Date.now();
                try {
                  const metrics = await deps.benchRunner.run(
                    createBenchInput(
                      state,
                      model.modelPath,
                      appliedRow,
                      promptTokens,
                      generationTokens,
                      capabilities,
                    ),
                  );
                  executed += 1;
                  const elapsed = (
                    (Date.now() - runStart) /
                    1e3
                  ).toFixed(1);
                  const record = buildSuccessRecord(
                    baseRecord,
                    metrics,
                  );
                  records.push(record);
                  await appendRunRecord(runsFilePath, record);
                  if (verbose) {
                    logger.log(
                      `  ${chalk.green("\u2713")} tg=${metrics.tgTokensPerSec.toFixed(1)} tok/s  pp=${metrics.ppTokensPerSec.toFixed(1)} tok/s  (${elapsed}s)`,
                    );
                  } else {
                    logger.log(
                      `[${runIndex}/${runCount}] tg=${metrics.tgTokensPerSec.toFixed(1)} tok/s  pp=${metrics.ppTokensPerSec.toFixed(1)} tok/s | ${configSummary}`,
                    );
                  }
                } catch (error) {
                  hadFailures = true;
                  failed += 1;
                  const elapsed = (
                    (Date.now() - runStart) /
                    1e3
                  ).toFixed(1);
                  const msg =
                    error instanceof Error
                      ? error.message
                      : String(error);
                  const record = buildFailedRecord(
                    baseRecord,
                    msg,
                  );
                  records.push(record);
                  await appendRunRecord(runsFilePath, record);
                  if (verbose) {
                    logger.log(
                      `  ${chalk.red("\u2717")} FAILED (${elapsed}s): ${msg}`,
                    );
                  } else {
                    logger.log(
                      `[${runIndex}/${runCount}] FAILED | ${configSummary}: ${msg}`,
                    );
                  }
                }
              }
            }
          }
        }

        if (verbose) {
          const totalElapsed = (
            (Date.now() - scanStartTime) /
            1e3
          ).toFixed(0);
          logger.log(
            chalk.bold("\n\u2500\u2500 Scan Complete \u2500\u2500"),
          );
          logger.log(
            `  Executed: ${executed}  Cached: ${cacheHits}  Failed: ${failed}  Total: ${runCount}`,
          );
          logger.log(`  Wall time: ${totalElapsed}s`);
        }
        const summaries = summarizeRecords(records);
        printScanSummary(logger, summaries, {
          modelName: model.modelName,
        });
        if (hadFailures) {
          process.exitCode = 1;
        }
      },
    );
}
