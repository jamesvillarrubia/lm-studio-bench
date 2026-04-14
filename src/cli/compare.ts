import { writeFile } from "fs/promises";
import path from "path";
import type { Command } from "commander";
import { fileExists, readJson, readYaml } from "../config/loader.js";
import {
  resolveDataRoot,
  ensureDataRoot,
  statePath,
  sweepConfigPath,
  modelRunsPath,
  modelSummaryPath,
  comparisonCsvPath,
} from "../config/data-root.js";
import {
  LocalStateSchema,
  SweepConfigSchema,
  RunRecordSchema,
} from "../config/schema.js";
import type { LocalState, RunRecord } from "../config/schema.js";
import {
  applyModelOverrides,
  filterToAppliedBenchmarkConfig,
  findUnsupportedBaselineKeys,
} from "../config/resolution.js";
import { probeLlamaBenchCapabilities } from "../runner/llama-bench-capabilities.js";
import type { BenchCapabilities } from "../runner/llama-bench-capabilities.js";
import { inferLlamaBenchBinaryFingerprint } from "../runner/bench-binary-fingerprint.js";
import { createBenchInput } from "../runner/llama-bench.js";
import type { BenchRunner } from "../runner/llama-bench.js";
import { computeRunKey } from "../cache/run-key.js";
import {
  appendRunRecord,
  loadRunRecords,
  findLatestRunRecordByRunKey,
  summarizeRecords,
} from "../reporter/results.js";
import {
  buildModelSummaryCsv,
  buildComparisonCsv,
} from "../reporter/csv.js";
import type { ComparisonRow } from "../reporter/csv.js";
import { printModelComparison } from "../reporter/terminal.js";
import type { LoggerLike } from "../reporter/terminal.js";

function collectString(value: string, previous: string[]): string[] {
  return [...previous, value];
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

function resolveModelPath(
  state: LocalState,
  modelRef: string,
  configModels: { name: string; path: string }[],
): { modelName: string; modelPath: string } {
  if (modelRef.includes("/") || modelRef.endsWith(".gguf")) {
    return {
      modelName: path.basename(modelRef, path.extname(modelRef)),
      modelPath: modelRef,
    };
  }
  const fromConfig = configModels.find((item) => item.name === modelRef);
  if (fromConfig) {
    return { modelName: fromConfig.name, modelPath: fromConfig.path };
  }
  const fromState = state.models.find((item) => item.name === modelRef);
  if (fromState) {
    return { modelName: fromState.name, modelPath: fromState.path };
  }
  throw new Error(`Model '${modelRef}' could not be resolved`);
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

export interface CompareDeps {
  dataRoot?: string;
  logger?: LoggerLike;
  now?: () => Date;
  benchRunner: BenchRunner;
  benchCapabilities?: BenchCapabilities;
}

export function registerCompareCommand(
  command: Command,
  deps: CompareDeps,
): void {
  const logger = deps.logger ?? console;
  const now = deps.now ?? (() => new Date());

  command
    .command("compare")
    .description(
      "Benchmark models head-to-head on baseline configuration",
    )
    .requiredOption(
      "-m, --model <model>",
      "Model name or path (repeat for multiple models)",
      collectString,
      [],
    )
    .option("-c, --config <path>", "Config path (default: <data-dir>/sweep-config.yaml)")
    .option(
      "--data-dir <path>",
      "Data directory (default: ~/.lmstudio-bench)",
    )
    .option(
      "--rerun",
      "Ignore successful benchmark cache hits and execute all planned runs",
    )
    .option(
      "--retry-failed",
      "Skip successful cache hits, but rerun failures (based on latest record per run key)",
    )
    .action(
      async (options: {
        model: string[];
        config?: string;
        dataDir?: string;
        rerun?: boolean;
        retryFailed?: boolean;
      }) => {
        if (options.model.length < 2) {
          throw new Error(
            "Provide at least two --model values for comparison",
          );
        }
        const root = resolveDataRoot(options.dataDir ?? deps.dataRoot);
        await ensureDataRoot(root);
        const sp = statePath(root);
        if (!(await fileExists(sp))) {
          throw new Error("Run `lmstudio-bench setup` first");
        }
        const configPath = sweepConfigPath(root, options.config);
        const state = await readJson(sp, LocalStateSchema);
        const capabilities =
          deps.benchCapabilities ??
          (await probeLlamaBenchCapabilities(state.llama_bench_path));
        for (const warning of capabilities.warnings) {
          logger.log(`llama-bench capabilities: ${warning}`);
        }
        const config = await readYaml(configPath, SweepConfigSchema);
        const runnerIdentity =
          (await deps.benchRunner.getRunnerIdentity?.({
            llamaBenchPath: state.llama_bench_path,
          })) ??
          (await inferLlamaBenchBinaryFingerprint(
            state.llama_bench_path,
          ));
        const rows: ComparisonRow[] = [];
        let hadFailures = false;

        for (const modelRef of options.model) {
          const model = resolveModelPath(
            state,
            modelRef,
            config.models,
          );
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
              `Ignoring unsupported baseline fields for ${model.modelName}: ${unsupportedBaselineKeys.join(", ")}`,
            );
          }
          const promptTokens =
            resolvedConfig.workload.prompt_tokens[0];
          const generationTokens =
            resolvedConfig.workload.generation_tokens[0];
          if (
            typeof promptTokens !== "number" ||
            typeof generationTokens !== "number"
          ) {
            throw new Error(
              "Workload prompt/generation tokens are required",
            );
          }
          const appliedBaseline = filterToAppliedBenchmarkConfig(
            resolvedConfig.baseline,
            capabilities,
          );
          const requestedBaselineRecord = tuningSnapshotToRecord(
            resolvedConfig.baseline,
          );
          const appliedBaselineRecord =
            tuningSnapshotToRecord(appliedBaseline);
          const runsPath = modelRunsPath(
            root,
            sanitizeFilePart(model.modelName),
          );
          const historicalRecords =
            await loadRunRecords(runsPath);
          const modelRecords: RunRecord[] = [];

          for (
            let repetition = 1;
            repetition <= resolvedConfig.workload.repetitions;
            repetition += 1
          ) {
            const runKey = computeRunKey({
              benchCommand: "compare",
              modelPath: model.modelPath,
              appliedConfig: appliedBaseline,
              workload: {
                pp: promptTokens,
                tg: generationTokens,
              },
              repetition: {
                runIndex: repetition,
                repetitions: resolvedConfig.workload.repetitions,
              },
              hardware: state.hardware as unknown as Record<
                string,
                unknown
              >,
              runnerIdentity,
            });
            const base = {
              id: `run-${now().toISOString().replace(/[:.]/g, "-")}-${repetition}`,
              timestamp: now().toISOString(),
              model: model.modelName,
              run_key: runKey,
              bench_command: "compare" as const,
              runner_identity: runnerIdentity,
              hardware: state.hardware,
              requested_config: requestedBaselineRecord,
              applied_config: appliedBaselineRecord,
              config: appliedBaselineRecord,
              capability_meta: {
                supported_keys: [...capabilities.supportedKeys].sort(),
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
              repetitions: resolvedConfig.workload.repetitions,
            };
            const latest = findLatestRunRecordByRunKey(
              [...historicalRecords, ...modelRecords],
              runKey,
            );
            const rerun = Boolean(options.rerun);
            const retryFailed = Boolean(options.retryFailed);
            const canReuseLatestSuccess =
              !rerun && latest?.status === "success";

            if (canReuseLatestSuccess) {
              const record = buildCacheHitRecord(base, latest);
              modelRecords.push(record);
              logger.log(
                `Cache hit: skipping benchmark${retryFailed ? " (--retry-failed)" : ""} (model=${model.modelName}, run_key=${runKey}, reused_run_id=${latest.id})`,
              );
              continue;
            }
            try {
              const metrics = await deps.benchRunner.run(
                createBenchInput(
                  state,
                  model.modelPath,
                  appliedBaseline,
                  promptTokens,
                  generationTokens,
                  capabilities,
                ),
              );
              const record = RunRecordSchema.parse({
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
              modelRecords.push(record);
              await appendRunRecord(runsPath, record);
            } catch (error) {
              hadFailures = true;
              const record = RunRecordSchema.parse({
                ...base,
                metrics: {
                  pp_tokens_per_sec: null,
                  tg_tokens_per_sec: null,
                  ttft_ms: null,
                  wall_time_sec: null,
                  peak_memory_mb: null,
                  load_time_ms: null,
                },
                status: "failed",
                error: {
                  message:
                    error instanceof Error
                      ? error.message
                      : String(error),
                },
              });
              modelRecords.push(record);
              await appendRunRecord(runsPath, record);
            }
          }
          const summaries = summarizeRecords(modelRecords);
          if (summaries.length === 0) {
            continue;
          }
          const summaryCsv = buildModelSummaryCsv(summaries);
          await writeFile(
            modelSummaryPath(root, sanitizeFilePart(model.modelName)),
            summaryCsv,
            "utf8",
          );
          const best = summaries[0];
          if (!best) {
            continue;
          }
          rows.push({
            model: model.modelName,
            bestMedianTg: best.medianTg,
            bestMedianPp: best.medianPp,
            bestMedianTtft: best.medianTtft,
            bestConfigKey: best.configKey,
          });
        }
        await writeFile(
          comparisonCsvPath(root),
          buildComparisonCsv(rows),
          "utf8",
        );
        printModelComparison(logger, rows);
        if (hadFailures) {
          process.exitCode = 1;
        }
      },
    );
}
