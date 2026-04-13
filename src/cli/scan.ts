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
import type { SensitivityEntry } from "../sweep/sensitivity.js";
import {
  buildCoarseScan,
  rankAxisImpact,
  buildPhase1Winner,
  buildFocusedGrid,
} from "../sweep/adaptive.js";
import type { PhaseResult, AxisImpact } from "../sweep/adaptive.js";
import { getDefaultAxisHints, applyAxisHints } from "../sweep/axis-hints.js";
import type { AxisHintMap } from "../sweep/axis-hints.js";
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

export type ScanStrategy = "oat" | "adaptive";

interface RunEntriesContext {
  entries: SensitivityEntry[];
  workload: { prompt_tokens: number[]; generation_tokens: number[]; repetitions: number };
  model: { modelName: string; modelPath: string };
  state: LocalState;
  capabilities: BenchCapabilities;
  runsFilePath: string;
  historicalRecords: RunRecord[];
  runnerIdentity: string;
  benchRunner: BenchRunner;
  now: () => Date;
  logger: LoggerLike;
  verbose: boolean;
  rerun: boolean;
  retryFailed: boolean;
  /** Offset for run numbering (for multi-phase scans). */
  runOffset?: number;
  totalOverride?: number;
  phaseLabel?: string;
}

interface RunEntriesResult {
  records: RunRecord[];
  phaseResults: PhaseResult[];
  executed: number;
  cacheHits: number;
  failed: number;
  hadFailures: boolean;
}

async function runEntries(ctx: RunEntriesContext): Promise<RunEntriesResult> {
  const records: RunRecord[] = [];
  const phaseResults: PhaseResult[] = [];
  let runIndex = ctx.runOffset ?? 0;
  let hadFailures = false;
  let cacheHits = 0;
  let executed = 0;
  let failed = 0;

  const totalRuns =
    ctx.totalOverride ??
    ctx.entries.length *
      ctx.workload.prompt_tokens.length *
      ctx.workload.generation_tokens.length *
      ctx.workload.repetitions;

  for (const entry of ctx.entries) {
    for (const promptTokens of ctx.workload.prompt_tokens) {
      for (const generationTokens of ctx.workload.generation_tokens) {
        for (
          let repetition = 1;
          repetition <= ctx.workload.repetitions;
          repetition += 1
        ) {
          runIndex += 1;
          const appliedRow = filterToAppliedBenchmarkConfig(
            entry.config,
            ctx.capabilities,
          );
          const runKey = computeRunKey({
            benchCommand: "scan",
            modelPath: ctx.model.modelPath,
            appliedConfig: appliedRow,
            workload: { pp: promptTokens, tg: generationTokens },
            repetition: {
              runIndex: repetition,
              repetitions: ctx.workload.repetitions,
            },
            hardware: ctx.state.hardware as unknown as Record<string, unknown>,
            runnerIdentity: ctx.runnerIdentity,
          });
          const requestedRecord = tuningSnapshotToRecord(entry.config);
          const appliedRecord = tuningSnapshotToRecord(appliedRow);
          const configSummary = Object.entries(appliedRecord)
            .map(([k, v]) => `${k}=${v}`)
            .join(" ");
          const label = ctx.phaseLabel ? `${ctx.phaseLabel} ` : "";
          const progressPrefix = `${label}[${runIndex}/${totalRuns}]`;

          const baseRecord = {
            id: `run-${ctx.now().toISOString().replace(/[:.]/g, "-")}-${runIndex}`,
            timestamp: ctx.now().toISOString(),
            model: ctx.model.modelName,
            run_key: runKey,
            bench_command: "scan" as const,
            runner_identity: ctx.runnerIdentity,
            hardware: ctx.state.hardware,
            requested_config: requestedRecord,
            applied_config: appliedRecord,
            config: appliedRecord,
            capability_meta: {
              supported_keys: [...ctx.capabilities.supportedKeys].sort(),
              used_fallback: ctx.capabilities.usedFallback,
              warnings:
                ctx.capabilities.warnings.length > 0
                  ? [...ctx.capabilities.warnings]
                  : undefined,
            },
            workload: { pp: promptTokens, tg: generationTokens },
            run_index: repetition,
            repetitions: ctx.workload.repetitions,
          };

          const latest = findLatestRunRecordByRunKey(
            [...ctx.historicalRecords, ...records],
            runKey,
          );
          const canReuseLatestSuccess =
            !ctx.rerun && latest?.status === "success";

          if (canReuseLatestSuccess) {
            cacheHits += 1;
            const record = buildCacheHitRecord(baseRecord, latest);
            records.push(record);
            phaseResults.push({
              variedParam: entry.variedParam,
              config: entry.config,
              tgTokensPerSec: latest.metrics.tg_tokens_per_sec,
              ppTokensPerSec: latest.metrics.pp_tokens_per_sec,
            });
            if (ctx.verbose) {
              ctx.logger.log(
                `${chalk.dim(progressPrefix)} ${chalk.yellow("CACHED")} pp=${promptTokens} tg=${generationTokens} | ${configSummary}`,
              );
            }
            continue;
          }

          if (ctx.verbose) {
            ctx.logger.log(
              `${chalk.cyan(progressPrefix)} ${chalk.bold("RUNNING")} pp=${promptTokens} tg=${generationTokens} rep=${repetition} | ${configSummary}`,
            );
          }

          const runStart = Date.now();
          try {
            const metrics = await ctx.benchRunner.run(
              createBenchInput(
                ctx.state,
                ctx.model.modelPath,
                appliedRow,
                promptTokens,
                generationTokens,
                ctx.capabilities,
              ),
            );
            executed += 1;
            const elapsed = ((Date.now() - runStart) / 1e3).toFixed(1);
            const record = buildSuccessRecord(baseRecord, metrics);
            records.push(record);
            await appendRunRecord(ctx.runsFilePath, record);
            phaseResults.push({
              variedParam: entry.variedParam,
              config: entry.config,
              tgTokensPerSec: metrics.tgTokensPerSec,
              ppTokensPerSec: metrics.ppTokensPerSec,
            });
            if (ctx.verbose) {
              ctx.logger.log(
                `  ${chalk.green("\u2713")} tg=${metrics.tgTokensPerSec.toFixed(1)} tok/s  pp=${metrics.ppTokensPerSec.toFixed(1)} tok/s  (${elapsed}s)`,
              );
            } else {
              ctx.logger.log(
                `${progressPrefix} tg=${metrics.tgTokensPerSec.toFixed(1)} tok/s  pp=${metrics.ppTokensPerSec.toFixed(1)} tok/s | ${configSummary}`,
              );
            }
          } catch (error) {
            hadFailures = true;
            failed += 1;
            const elapsed = ((Date.now() - runStart) / 1e3).toFixed(1);
            const msg =
              error instanceof Error ? error.message : String(error);
            const record = buildFailedRecord(baseRecord, msg);
            records.push(record);
            await appendRunRecord(ctx.runsFilePath, record);
            phaseResults.push({
              variedParam: entry.variedParam,
              config: entry.config,
              tgTokensPerSec: null,
              ppTokensPerSec: null,
            });
            if (ctx.verbose) {
              ctx.logger.log(
                `  ${chalk.red("\u2717")} FAILED (${elapsed}s): ${msg}`,
              );
            } else {
              ctx.logger.log(
                `${progressPrefix} FAILED | ${configSummary}: ${msg}`,
              );
            }
          }
        }
      }
    }
  }

  return { records, phaseResults, executed, cacheHits, failed, hadFailures };
}

function printImpactReport(
  logger: LoggerLike,
  impacts: AxisImpact[],
): void {
  logger.log(chalk.bold("\n── Axis Impact (phase 1) ──"));
  for (const impact of impacts) {
    const values = impact.valueResults
      .map(
        (v) =>
          `${v.value}=${v.medianTg.toFixed(1)}`,
      )
      .join(", ");
    logger.log(
      `  ${impact.axis}: range=${impact.tgRange.toFixed(1)} tok/s  best=${String(impact.bestValue)}  [${values}]`,
    );
  }
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
    .option(
      "-s, --strategy <strategy>",
      "Sweep strategy: oat (one-at-a-time) or adaptive (two-phase)",
      "oat",
    )
    .action(
      async (options: {
        model?: string;
        config?: string;
        estimateOnly?: boolean;
        rerun?: boolean;
        retryFailed?: boolean;
        verbose?: boolean;
        strategy?: string;
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

        const strategy: ScanStrategy =
          options.strategy === "adaptive" ? "adaptive" : "oat";
        const baseline = resolvedConfig.baseline as Record<
          string,
          string | number | boolean
        >;

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

        const scanStartTime = Date.now();
        let allRecords: RunRecord[] = [];
        let totalExecuted = 0;
        let totalCacheHits = 0;
        let totalFailed = 0;
        let hadFailures = false;

        const sharedCtx = {
          workload: resolvedConfig.workload,
          model,
          state,
          capabilities,
          runsFilePath,
          historicalRecords,
          runnerIdentity,
          benchRunner: deps.benchRunner,
          now,
          logger,
          verbose,
          rerun: Boolean(options.rerun),
          retryFailed: Boolean(options.retryFailed),
        };

        if (strategy === "adaptive") {
          // ── Apply axis hints to pin known-optimal values ──
          // Hardware defaults, then YAML overrides on top
          const yamlHints = config.axis_hints ?? {};
          const hints: AxisHintMap = {
            ...getDefaultAxisHints(state.hardware),
            ...Object.fromEntries(
              Object.entries(yamlHints).map(([k, v]) => [
                k,
                {
                  direction: v.direction,
                  reason: v.reason ?? "User override in sweep-config.yaml",
                },
              ]),
            ),
          };
          const { pinnedOverrides, reducedScan, pinLog } =
            applyAxisHints(appliedScan, hints);

          // Override baseline with pinned values
          const adaptiveBaseline = { ...baseline, ...pinnedOverrides };

          if (verbose && pinLog.length > 0) {
            logger.log(
              chalk.bold("\n── Pinned Axes (known-optimal) ──"),
            );
            for (const pin of pinLog) {
              logger.log(
                `  ${chalk.green("✓")} ${pin.axis}=${String(pin.value)} — ${pin.reason}`,
              );
            }
          } else if (pinLog.length > 0) {
            logger.log(
              `Pinned ${pinLog.length} axis(es) to known-optimal: ${pinLog.map((p) => `${p.axis}=${String(p.value)}`).join(", ")}`,
            );
          }

          // ── Phase 1: Coarse OAT on remaining uncertain axes ──
          const coarseScan = buildCoarseScan(reducedScan, 4);
          const phase1Entries = expandSensitivityScan(
            adaptiveBaseline,
            coarseScan,
          );
          const phase1Runs =
            phase1Entries.length *
            resolvedConfig.workload.prompt_tokens.length *
            resolvedConfig.workload.generation_tokens.length *
            resolvedConfig.workload.repetitions;

          if (verbose) {
            logger.log(
              chalk.bold(
                "\n── Phase 1: Coarse OAT ──",
              ),
            );
            const coarseAxes = Object.entries(coarseScan).filter(
              ([, v]) =>
                Array.isArray(v) && (v as unknown[]).length > 0,
            );
            for (const [axis, values] of coarseAxes) {
              logger.log(
                `  ${axis}: [${(values as unknown[]).join(", ")}]`,
              );
            }
            logger.log(
              `  ${phase1Entries.length} configs × ${resolvedConfig.workload.repetitions} reps = ${chalk.cyan(String(phase1Runs))} runs`,
            );
          }

          if (options.estimateOnly) {
            logger.log(
              `Estimated phase 1 runs: ${phase1Runs} (phase 2 depends on results)`,
            );
            return;
          }

          const phase1 = await runEntries({
            ...sharedCtx,
            entries: phase1Entries,
            phaseLabel: "P1",
          });

          allRecords.push(...phase1.records);
          totalExecuted += phase1.executed;
          totalCacheHits += phase1.cacheHits;
          totalFailed += phase1.failed;
          if (phase1.hadFailures) hadFailures = true;

          // ── Analyze impact ──
          const impacts = rankAxisImpact(phase1.phaseResults);
          if (verbose || impacts.length > 0) {
            printImpactReport(logger, impacts);
          }

          const significantImpacts = impacts.filter(
            (i) => i.tgRange > 0.5,
          );

          if (significantImpacts.length === 0) {
            logger.log(
              chalk.yellow(
                "\nNo significant axis impact detected — skipping phase 2.",
              ),
            );
          } else {
            // ── Phase 2: Focused grid ──
            const winner = buildPhase1Winner(
              adaptiveBaseline,
              significantImpacts,
            );
            const phase2Entries = buildFocusedGrid(
              winner,
              significantImpacts,
              reducedScan,
              3,
              5,
            );
            const phase2Runs =
              phase2Entries.length *
              resolvedConfig.workload.prompt_tokens.length *
              resolvedConfig.workload.generation_tokens.length *
              resolvedConfig.workload.repetitions;

            if (verbose) {
              logger.log(
                chalk.bold(
                  "\n── Phase 2: Focused Grid ──",
                ),
              );
              logger.log(
                `  Top axes: ${significantImpacts.slice(0, 3).map((i) => i.axis).join(", ")}`,
              );
              logger.log(
                `  Winner baseline: ${JSON.stringify(winner)}`,
              );
              logger.log(
                `  ${phase2Entries.length} configs × ${resolvedConfig.workload.repetitions} reps = ${chalk.cyan(String(phase2Runs))} runs`,
              );
            }

            const phase2 = await runEntries({
              ...sharedCtx,
              entries: phase2Entries,
              historicalRecords: [
                ...historicalRecords,
                ...allRecords,
              ],
              runOffset: phase1Runs,
              phaseLabel: "P2",
            });

            allRecords.push(...phase2.records);
            totalExecuted += phase2.executed;
            totalCacheHits += phase2.cacheHits;
            totalFailed += phase2.failed;
            if (phase2.hadFailures) hadFailures = true;
          }
        } else {
          // ── Standard OAT ──
          const entries = expandSensitivityScan(
            baseline,
            appliedScan,
          );
          const runCount =
            entries.length *
            resolvedConfig.workload.prompt_tokens.length *
            resolvedConfig.workload.generation_tokens.length *
            resolvedConfig.workload.repetitions;

          if (verbose) {
            logger.log(
              chalk.bold("\n── Run Estimate ──"),
            );
            logger.log(
              `  ${entries.length} configs × ${resolvedConfig.workload.prompt_tokens.length} prompt sizes × ${resolvedConfig.workload.generation_tokens.length} gen sizes × ${resolvedConfig.workload.repetitions} reps = ${chalk.cyan(String(runCount))} total runs`,
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

          const result = await runEntries({
            ...sharedCtx,
            entries,
          });

          allRecords = result.records;
          totalExecuted = result.executed;
          totalCacheHits = result.cacheHits;
          totalFailed = result.failed;
          hadFailures = result.hadFailures;
        }

        if (verbose) {
          const totalElapsed = (
            (Date.now() - scanStartTime) /
            1e3
          ).toFixed(0);
          logger.log(
            chalk.bold("\n── Scan Complete ──"),
          );
          logger.log(
            `  Strategy: ${strategy}  Executed: ${totalExecuted}  Cached: ${totalCacheHits}  Failed: ${totalFailed}`,
          );
          logger.log(`  Wall time: ${totalElapsed}s`);
        }
        const summaries = summarizeRecords(allRecords);
        printScanSummary(logger, summaries, {
          modelName: model.modelName,
        });
        if (hadFailures) {
          process.exitCode = 1;
        }
      },
    );
}
