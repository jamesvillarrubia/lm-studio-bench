import { describe, expect, test } from "vitest";
import type { RunRecord } from "../../src/config/schema.js";
import {
  analyzeFactorImportance,
} from "../../src/analysis/factor-importance.js";

function makeRecord(args: {
  id: string;
  runIndex: number;
  threads: number;
  mlock: boolean;
  nCtx: number;
  tg: number;
}): RunRecord {
  return {
    id: args.id,
    timestamp: "2026-04-14T00:00:00.000Z",
    model: "demo",
    run_key: args.id,
    bench_command: "scan",
    runner_identity: "runner",
    hardware: {
      chip: "Apple Silicon",
      cores: 12,
      p_cores: 8,
      e_cores: 4,
      memory_gb: 64,
      metal: true,
    },
    requested_config: {
      threads: args.threads,
      mlock: args.mlock,
      n_ctx: args.nCtx,
    },
    applied_config: {
      threads: args.threads,
      mlock: args.mlock,
      n_ctx: args.nCtx,
    },
    config: {
      threads: args.threads,
      mlock: args.mlock,
      n_ctx: args.nCtx,
    },
    workload: { pp: 512, tg: 128 },
    metrics: {
      pp_tokens_per_sec: 100,
      tg_tokens_per_sec: args.tg,
      ttft_ms: null,
      peak_memory_mb: null,
      load_time_ms: null,
      wall_time_sec: null,
    },
    run_index: args.runIndex,
    repetitions: 2,
    status: "success",
    error: null,
  };
}

describe("analyzeFactorImportance", () => {
  test("ranks the strongest factors highest and flags nonlinear numeric effects", () => {
    const records: RunRecord[] = [];
    let id = 0;
    const threadLevels = [2, 4, 8, 12];
    const ctxLevels = [2048, 4096, 8192];
    const mlockLevels = [false, true];

    for (const threads of threadLevels) {
      for (const nCtx of ctxLevels) {
        for (const mlock of mlockLevels) {
          for (let rep = 1; rep <= 2; rep += 1) {
            const tg =
              20 +
              threads * 2.2 +
              (mlock ? -1 : 0) +
              (nCtx === 4096 ? 6 : nCtx === 8192 ? 2 : 0) +
              rep * 0.1;
            id += 1;
            records.push(
              makeRecord({
                id: `r-${id}`,
                runIndex: rep,
                threads,
                mlock,
                nCtx,
                tg,
              }),
            );
          }
        }
      }
    }

    const report = analyzeFactorImportance(records, {
      responseMetric: "tg_tokens_per_sec",
      bootstrapIterations: 50,
      topInteractions: 2,
      seed: 123,
    });

    expect(report.factors.length).toBeGreaterThanOrEqual(3);
    expect(report.factors[0]?.factor).toBe("threads");
    expect(report.factors.find((f) => f.factor === "n_ctx")?.nonlinear).toBe(true);
    expect(report.factors.find((f) => f.factor === "mlock")?.importance).toBeGreaterThan(0);
    expect(report.factors[0]?.ci.lower).toBeLessThanOrEqual(
      report.factors[0]?.importance ?? 0,
    );
    expect(report.factors[0]?.ci.upper).toBeGreaterThanOrEqual(
      report.factors[0]?.importance ?? 0,
    );
  });
});
