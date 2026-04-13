import {
  BenchmarkBaselineSchema,
  BenchmarkScanSchema,
  WorkloadSchema,
} from "./schema.js";
import type { SweepConfig } from "./schema.js";
import {
  BENCHMARK_TUNE_KEYS,
} from "../runner/llama-bench-capabilities.js";
import type {
  BenchmarkTuneKey,
  BenchCapabilities,
} from "../runner/llama-bench-capabilities.js";

export interface ResolvedSweepConfig {
  baseline: Record<string, string | number | boolean>;
  scan: Record<string, (string | number | boolean)[]>;
  workload: {
    prompt_tokens: number[];
    generation_tokens: number[];
    repetitions: number;
  };
}

/**
 * Legacy list used by older code paths that don't yet do dynamic capability probing.
 * Includes `no_kv_offload` alongside the original set.
 */
export const SUPPORTED_SCAN_KEYS: readonly string[] = [
  "n_ctx",
  "n_batch",
  "n_ubatch",
  "n_gpu_layers",
  "threads",
  "threads_batch",
  "kv_type_key",
  "kv_type_value",
  "flash_attention",
  "mmap",
  "mlock",
  "no_kv_offload",
];

export function applyModelOverrides(
  config: SweepConfig,
  model: { name: string; path: string },
): ResolvedSweepConfig {
  const matchedOverrides = config.model_overrides.filter((override) => {
    const nameMatch = override.match.name
      ? override.match.name === model.name
      : true;
    const pathMatch = override.match.path
      ? override.match.path === model.path
      : true;
    return nameMatch && pathMatch;
  });

  let baseline = { ...config.baseline } as Record<string, unknown>;
  let scan = { ...config.scan } as Record<string, unknown>;
  let workload = { ...config.workload } as Record<string, unknown>;

  for (const override of matchedOverrides) {
    baseline = BenchmarkBaselineSchema.parse({
      ...baseline,
      ...(override.baseline ?? {}),
    });
    scan = BenchmarkScanSchema.parse({
      ...scan,
      ...(override.scan ?? {}),
    });
    workload = WorkloadSchema.parse({
      ...workload,
      ...(override.workload ?? {}),
    });
  }

  return {
    baseline: baseline as Record<string, string | number | boolean>,
    scan: scan as Record<string, (string | number | boolean)[]>,
    workload: workload as ResolvedSweepConfig["workload"],
  };
}

function isRecordKey(key: string): key is BenchmarkTuneKey {
  return key in BenchmarkBaselineSchema.shape;
}

export type BenchmarkConfig = Record<string, string | number | boolean>;

export function filterToAppliedBenchmarkConfig(
  source: Record<string, unknown>,
  capabilities: BenchCapabilities,
): BenchmarkConfig {
  const out: BenchmarkConfig = {};
  for (const key of BENCHMARK_TUNE_KEYS) {
    if (!capabilities.supportedKeys.has(key)) {
      continue;
    }
    if (typeof source === "object" && source !== null && key in source) {
      out[key] = source[key] as string | number | boolean;
    }
  }
  return out;
}

export function filterScanForCapabilities(
  scan: Record<string, unknown>,
  capabilities: BenchCapabilities,
): Record<string, (string | number | boolean)[]> {
  return Object.fromEntries(
    Object.entries(scan).filter(([key]) => {
      if (!isRecordKey(key)) {
        return false;
      }
      return capabilities.supportedKeys.has(key);
    }),
  ) as Record<string, (string | number | boolean)[]>;
}

export function findUnsupportedScanKeys(
  scan: Record<string, unknown>,
  capabilities: BenchCapabilities,
): string[] {
  return Object.keys(scan).filter((key) => {
    if (!isRecordKey(key)) {
      return false;
    }
    const values = scan[key] as unknown[];
    return (
      Array.isArray(values) &&
      values.length > 0 &&
      !capabilities.supportedKeys.has(key)
    );
  });
}

export function findUnsupportedBaselineKeys(
  baseline: Record<string, unknown>,
  capabilities: BenchCapabilities,
): string[] {
  return Object.keys(baseline).filter(
    (key) => isRecordKey(key) && !capabilities.supportedKeys.has(key),
  );
}

export function staticCapabilitiesFromLegacyList(): BenchCapabilities {
  return {
    supportedKeys: new Set(SUPPORTED_SCAN_KEYS),
    warnings: [],
    usedFallback: false,
  };
}
