import type { BenchmarkConfig, SensitivityEntry } from "./sensitivity.js";

/**
 * Classification of a sweep axis for search strategy selection.
 */
export type AxisKind = "numeric" | "categorical" | "boolean";

export function classifyAxis(
  values: (string | number | boolean)[],
): AxisKind {
  if (values.length === 0) return "categorical";
  if (values.every((v) => typeof v === "boolean")) return "boolean";
  if (values.every((v) => typeof v === "number")) return "numeric";
  return "categorical";
}

/**
 * For a sorted numeric array, pick the value at the given percentile (0-1).
 * Snaps to the nearest actual value in the array.
 */
function pickPercentile(sorted: number[], pct: number): number {
  const idx = Math.round(pct * (sorted.length - 1));
  return sorted[Math.min(idx, sorted.length - 1)]!;
}

/**
 * Pick ternary probe points (25th, 50th, 75th percentile) from a sorted
 * numeric array. Returns deduplicated values.
 */
export function ternaryProbePoints(sorted: number[]): number[] {
  if (sorted.length <= 3) return [...sorted];

  const lo = pickPercentile(sorted, 0.25);
  const mid = pickPercentile(sorted, 0.5);
  const hi = pickPercentile(sorted, 0.75);

  const points = new Set([lo, mid, hi]);
  // Always include endpoints so we see the full range
  points.add(sorted[0]!);
  points.add(sorted[sorted.length - 1]!);

  return [...points].sort((a, b) => a - b);
}

/**
 * After a ternary round, narrow the search range toward the best value.
 * Returns the subset of the full sorted values that fall within the
 * narrowed range, then picks new probe points within that range.
 */
export function narrowRange(
  fullSorted: number[],
  bestValue: number,
  round: number,
): number[] {
  if (fullSorted.length <= 3) return [...fullSorted];

  const bestIdx = fullSorted.indexOf(bestValue);
  if (bestIdx === -1) return ternaryProbePoints(fullSorted);

  // Each round halves the window
  const windowFraction = 1 / Math.pow(2, round);
  const halfWindow = Math.max(
    1,
    Math.floor((fullSorted.length * windowFraction) / 2),
  );

  const lo = Math.max(0, bestIdx - halfWindow);
  const hi = Math.min(fullSorted.length - 1, bestIdx + halfWindow);
  const narrowed = fullSorted.slice(lo, hi + 1);

  if (narrowed.length <= 3) return narrowed;
  return ternaryProbePoints(narrowed);
}

/**
 * Partition scan axes into their search strategies.
 */
export interface AxisPartition {
  /** Axes that get ternary search (numeric with 4+ values). */
  ternary: { axis: string; sortedValues: number[] }[];
  /** Axes tested exhaustively (categorical, boolean, small numeric). */
  exhaustive: { axis: string; values: (string | number | boolean)[] }[];
}

export function partitionAxes(
  scan: Record<string, (string | number | boolean)[]>,
): AxisPartition {
  const ternary: AxisPartition["ternary"] = [];
  const exhaustive: AxisPartition["exhaustive"] = [];

  for (const [axis, values] of Object.entries(scan)) {
    if (!values || values.length === 0) continue;

    const kind = classifyAxis(values);
    if (kind === "numeric" && values.length >= 4) {
      const sorted = (values as number[]).slice().sort((a, b) => a - b);
      ternary.push({ axis, sortedValues: sorted });
    } else {
      exhaustive.push({ axis, values });
    }
  }

  return { ternary, exhaustive };
}

/**
 * Build sensitivity entries for one round of ternary probing.
 * Each ternary axis gets its probe points tested against the current baseline.
 * Exhaustive axes get all their values tested.
 */
export function buildTernaryRoundEntries(
  baseline: BenchmarkConfig,
  partition: AxisPartition,
  probePoints: Map<string, number[]>,
): SensitivityEntry[] {
  const deduped = new Map<string, SensitivityEntry>();

  // Baseline entry
  const baseKey = JSON.stringify(baseline);
  deduped.set(baseKey, { variedParam: "baseline", config: baseline });

  // Ternary axes: test probe points
  for (const { axis } of partition.ternary) {
    const points = probePoints.get(axis);
    if (!points) continue;
    for (const value of points) {
      const config = { ...baseline, [axis]: value };
      const key = JSON.stringify(config);
      if (!deduped.has(key)) {
        deduped.set(key, { variedParam: axis, config });
      }
    }
  }

  // Exhaustive axes: test all values
  for (const { axis, values } of partition.exhaustive) {
    for (const value of values) {
      const config = { ...baseline, [axis]: value };
      const key = JSON.stringify(config);
      if (!deduped.has(key)) {
        deduped.set(key, { variedParam: axis, config });
      }
    }
  }

  return Array.from(deduped.values());
}

/**
 * Extract the best value for a given axis from phase results.
 */
export function bestValueForAxis(
  results: { variedParam: string; config: BenchmarkConfig; tgTokensPerSec: number | null }[],
  axis: string,
): number | null {
  let best: { value: number; tg: number } | null = null;

  for (const r of results) {
    if (r.variedParam !== axis || r.tgTokensPerSec === null) continue;
    const val = r.config[axis];
    if (typeof val !== "number") continue;
    if (best === null || r.tgTokensPerSec > best.tg) {
      best = { value: val, tg: r.tgTokensPerSec };
    }
  }

  return best?.value ?? null;
}
