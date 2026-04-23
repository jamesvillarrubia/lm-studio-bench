import type { BenchmarkConfig, SensitivityEntry } from "./sensitivity.js";

/**
 * Result of a single benchmark run used for impact analysis.
 */
export interface PhaseResult {
  variedParam: string;
  config: BenchmarkConfig;
  tgTokensPerSec: number | null;
  ppTokensPerSec: number | null;
}

/**
 * Measured impact of a single axis from OAT phase 1.
 */
export interface AxisImpact {
  axis: string;
  /** Absolute range of median tg throughput across values tested. */
  tgRange: number;
  /** The value that produced the best tg throughput. */
  bestValue: string | number | boolean;
  /** The median tg at baseline for this axis. */
  baselineTg: number;
  /** All values tested and their median tg. */
  valueResults: { value: string | number | boolean; medianTg: number }[];
}

export interface NonlinearAxis {
  axis: string;
  signChanges: number;
  range: number;
}

export interface CoverageStats {
  axis: string;
  targetPerLevel: number;
  levelCounts: { value: string | number | boolean; count: number }[];
}

/**
 * Given the full scan values, pick a coarse subset (up to `maxPerAxis` evenly
 * spaced values including the endpoints).
 */
export function coarsenValues(
  values: (string | number | boolean)[],
  maxPerAxis: number,
): (string | number | boolean)[] {
  if (values.length <= maxPerAxis) {
    return values;
  }

  // Booleans and strings: return all (they're typically small)
  if (values.every((v) => typeof v === "boolean" || typeof v === "string")) {
    return values;
  }

  // Numeric: pick evenly spaced including endpoints
  const numericValues = values.filter(
    (v): v is number => typeof v === "number",
  );
  if (numericValues.length <= maxPerAxis) {
    return values;
  }

  const sorted = [...numericValues].sort((a, b) => a - b);
  const result: number[] = [];
  for (let i = 0; i < maxPerAxis; i++) {
    const idx = Math.round((i * (sorted.length - 1)) / (maxPerAxis - 1));
    const val = sorted[idx];
    if (val !== undefined && !result.includes(val)) {
      result.push(val);
    }
  }
  return result;
}

/**
 * Build a coarsened OAT scan for phase 1. Takes the full scan config and
 * reduces each axis to at most `maxPerAxis` values.
 */
export function buildCoarseScan(
  scan: Record<string, (string | number | boolean)[]>,
  maxPerAxis = 4,
): Record<string, (string | number | boolean)[]> {
  const coarse: Record<string, (string | number | boolean)[]> = {};
  for (const [key, values] of Object.entries(scan)) {
    if (!values || values.length === 0) {
      coarse[key] = [];
      continue;
    }
    coarse[key] = coarsenValues(values, maxPerAxis);
  }
  return coarse;
}

/**
 * Compute per-axis impact from phase 1 OAT results.
 * Groups results by variedParam, computes median tg for each value,
 * and ranks axes by throughput range (max - min median tg).
 */
export function rankAxisImpact(results: PhaseResult[]): AxisImpact[] {
  // Group by varied param
  const byAxis = new Map<string, PhaseResult[]>();
  for (const r of results) {
    if (r.variedParam === "baseline" || r.tgTokensPerSec === null) {
      continue;
    }
    const group = byAxis.get(r.variedParam) ?? [];
    group.push(r);
    byAxis.set(r.variedParam, group);
  }

  const impacts: AxisImpact[] = [];

  for (const [axis, group] of byAxis.entries()) {
    // Group by the value of the varied param
    const byValue = new Map<string, number[]>();
    for (const r of group) {
      const val = r.config[axis];
      if (val === undefined || r.tgTokensPerSec === null) continue;
      const key = String(val);
      const arr = byValue.get(key) ?? [];
      arr.push(r.tgTokensPerSec);
      byValue.set(key, arr);
    }

    const valueResults: {
      value: string | number | boolean;
      medianTg: number;
    }[] = [];
    for (const [valStr, tgValues] of byValue.entries()) {
      const med = medianOf(tgValues);
      if (med === null) continue;
      // Recover original type
      const originalVal = group.find(
        (r) => String(r.config[axis]) === valStr,
      )?.config[axis];
      valueResults.push({ value: originalVal ?? valStr, medianTg: med });
    }

    if (valueResults.length < 2) continue;

    const tgValues = valueResults.map((v) => v.medianTg);
    const tgRange = Math.max(...tgValues) - Math.min(...tgValues);
    const bestEntry = valueResults.reduce((a, b) =>
      b.medianTg > a.medianTg ? b : a,
    );

    impacts.push({
      axis,
      tgRange,
      bestValue: bestEntry.value,
      baselineTg: bestEntry.medianTg,
      valueResults,
    });
  }

  return impacts.sort((a, b) => b.tgRange - a.tgRange);
}

/**
 * Detect non-monotonic numeric axes from impact value curves.
 * We count direction changes in adjacent median deltas (ignoring tiny deltas).
 */
export function detectNonlinearNumericAxes(
  impacts: AxisImpact[],
  epsilon = 0.25,
): NonlinearAxis[] {
  const nonlinear: NonlinearAxis[] = [];

  for (const impact of impacts) {
    const numericPoints = impact.valueResults
      .filter(
        (
          v,
        ): v is {
          value: number;
          medianTg: number;
        } => typeof v.value === "number",
      )
      .sort((a, b) => a.value - b.value);

    if (numericPoints.length < 4) continue;

    let signChanges = 0;
    let prevSign = 0;
    for (let i = 1; i < numericPoints.length; i += 1) {
      const left = numericPoints[i - 1];
      const right = numericPoints[i];
      if (!left || !right) continue;
      const delta = right.medianTg - left.medianTg;
      if (Math.abs(delta) < epsilon) continue;
      const sign = delta > 0 ? 1 : -1;
      if (prevSign !== 0 && sign !== prevSign) {
        signChanges += 1;
      }
      prevSign = sign;
    }

    if (signChanges > 0) {
      nonlinear.push({
        axis: impact.axis,
        signChanges,
        range: impact.tgRange,
      });
    }
  }

  return nonlinear.sort((a, b) => {
    if (b.signChanges !== a.signChanges) return b.signChanges - a.signChanges;
    return b.range - a.range;
  });
}

/**
 * Build extra one-axis refinement configs for selected nonlinear axes,
 * using the full axis value list (not coarse/ternary probes).
 */
export function buildNonlinearRefinementEntries(
  baseline: BenchmarkConfig,
  axes: string[],
  fullScan: Record<string, (string | number | boolean)[]>,
): SensitivityEntry[] {
  const deduped = new Map<string, SensitivityEntry>();
  deduped.set(JSON.stringify(baseline), {
    variedParam: "nonlinear_baseline",
    config: baseline,
  });

  for (const axis of axes) {
    const values = fullScan[axis] ?? [];
    for (const value of values) {
      const cfg = { ...baseline, [axis]: value };
      const key = JSON.stringify(cfg);
      if (!deduped.has(key)) {
        deduped.set(key, { variedParam: axis, config: cfg });
      }
    }
  }

  return Array.from(deduped.values());
}

/**
 * Build DOE-style balancing entries to improve per-level coverage.
 * We use a budgeted round-robin over underrepresented levels so sparse levels
 * (e.g. n_cpu_moe=4) get additional samples without exploding run count.
 */
export function buildCoverageBalancingEntries(
  baseline: BenchmarkConfig,
  scan: Record<string, (string | number | boolean)[]>,
  results: PhaseResult[],
  repetitions: number,
  maxConfigs = 24,
  minPerLevelFloor = 6,
): { entries: SensitivityEntry[]; stats: CoverageStats[] } {
  const byAxisCounts = new Map<string, Map<string, number>>();

  for (const r of results) {
    if (r.tgTokensPerSec === null) continue;
    for (const [axis, values] of Object.entries(scan)) {
      if (!values || values.length === 0) continue;
      const val = r.config[axis];
      if (val === undefined) continue;
      let axisMap = byAxisCounts.get(axis);
      if (!axisMap) {
        axisMap = new Map<string, number>();
        byAxisCounts.set(axis, axisMap);
      }
      const key = String(val);
      axisMap.set(key, (axisMap.get(key) ?? 0) + 1);
    }
  }

  const deficits = new Map<string, Map<string, number>>();
  const axisValueLookup = new Map<string, Map<string, string | number | boolean>>();
  const stats: CoverageStats[] = [];

  for (const [axis, values] of Object.entries(scan)) {
    if (!values || values.length <= 1) continue;

    const counts = byAxisCounts.get(axis) ?? new Map<string, number>();
    const valueCountRows = values.map((value) => ({
      value,
      key: String(value),
      count: counts.get(String(value)) ?? 0,
    }));
    const sortedCounts = valueCountRows
      .map((v) => v.count)
      .sort((a, b) => a - b);
    const medianCount =
      sortedCounts.length > 0
        ? sortedCounts[Math.floor(sortedCounts.length / 2)] ?? 0
        : 0;
    const minPerLevel = Math.max(repetitions * 3, minPerLevelFloor);
    const targetPerLevel = Math.max(minPerLevel, medianCount);

    const axisDeficits = new Map<string, number>();
    const valueMap = new Map<string, string | number | boolean>();
    for (const row of valueCountRows) {
      valueMap.set(row.key, row.value);
      const deficit = Math.max(0, targetPerLevel - row.count);
      if (deficit > 0) {
        axisDeficits.set(row.key, deficit);
      }
    }
    axisValueLookup.set(axis, valueMap);
    deficits.set(axis, axisDeficits);
    stats.push({
      axis,
      targetPerLevel,
      levelCounts: valueCountRows.map((r) => ({ value: r.value, count: r.count })),
    });
  }

  const entries: SensitivityEntry[] = [];
  const seen = new Set<string>();
  const activeAxes = Array.from(deficits.keys());

  // Deterministic pseudo-random generator for reproducible DOE samples.
  let rngState = 0x9e3779b9;
  function rand(): number {
    rngState ^= rngState << 13;
    rngState ^= rngState >>> 17;
    rngState ^= rngState << 5;
    // Keep in uint32 space and map to [0,1)
    return ((rngState >>> 0) % 1_000_000) / 1_000_000;
  }

  function weightedPick<T>(items: { value: T; weight: number }[]): T | null {
    const filtered = items.filter((i) => i.weight > 0);
    if (filtered.length === 0) return null;
    const total = filtered.reduce((sum, i) => sum + i.weight, 0);
    let r = rand() * total;
    for (const item of filtered) {
      r -= item.weight;
      if (r <= 0) return item.value;
    }
    return filtered[filtered.length - 1]?.value ?? null;
  }

  let attempts = 0;
  const maxAttempts = Math.max(maxConfigs * 20, 200);
  while (entries.length < maxConfigs && attempts < maxAttempts) {
    attempts += 1;
    const config: BenchmarkConfig = { ...baseline };

    // 1) Pick a primary under-covered axis (weighted by total deficit)
    const axisPick = weightedPick(
      activeAxes.map((axis) => {
        const axisDeficits = deficits.get(axis);
        const totalDeficit = axisDeficits
          ? Array.from(axisDeficits.values()).reduce(
              (sum, d) => sum + Math.max(0, d),
              0,
            )
          : 0;
        return { value: axis, weight: totalDeficit };
      }),
    );
    if (!axisPick) break;

    // 2) For each axis, sample a level. Primary axis is strongly deficit-weighted.
    for (const axis of activeAxes) {
      const values = scan[axis] ?? [];
      if (values.length === 0) continue;
      const axisDeficits = deficits.get(axis);
      const picked = weightedPick(
        values.map((v) => {
          const d = axisDeficits?.get(String(v)) ?? 0;
          // Primary axis aggressively chases deficits; others still explore.
          const base = axis === axisPick ? 1 : 0.35;
          const weight = base + Math.max(0, d);
          return { value: v, weight };
        }),
      );
      if (picked !== null) {
        config[axis] = picked;
      }
    }

    const key = JSON.stringify(config);
    if (seen.has(key)) continue;
    seen.add(key);
    entries.push({ variedParam: "coverage_doe", config });

    // 3) Update deficit accounting after selecting this config
    for (const axis of activeAxes) {
      const axisDeficits = deficits.get(axis);
      if (!axisDeficits) continue;
      const keyForAxis = String(config[axis]);
      const cur = axisDeficits.get(keyForAxis);
      if (typeof cur === "number" && cur > 0) {
        axisDeficits.set(keyForAxis, Math.max(0, cur - repetitions));
      }
    }
  }

  return { entries, stats };
}

/**
 * Build the "winner" config from phase 1 — take the baseline and override
 * each axis with its best-performing value.
 */
export function buildPhase1Winner(
  baseline: BenchmarkConfig,
  impacts: AxisImpact[],
): BenchmarkConfig {
  const winner = { ...baseline };
  for (const impact of impacts) {
    winner[impact.axis] = impact.bestValue;
  }
  return winner;
}

/**
 * Build phase 2 focused grid: for the top N most impactful axes,
 * generate a dense local grid around the phase 1 winner.
 *
 * For numeric axes, generates values between the two best phase 1 values.
 * For boolean/string axes, includes all values.
 */
export function buildFocusedGrid(
  winner: BenchmarkConfig,
  impacts: AxisImpact[],
  fullScan: Record<string, (string | number | boolean)[]>,
  topN = 3,
  maxPerAxis = 5,
): SensitivityEntry[] {
  const focusAxes = impacts.slice(0, topN);
  const deduped = new Map<string, SensitivityEntry>();

  // Start with the winner as baseline
  const winnerEntry: SensitivityEntry = {
    variedParam: "phase2_baseline",
    config: winner,
  };
  deduped.set(JSON.stringify(winner), winnerEntry);

  for (const impact of focusAxes) {
    const fullValues = fullScan[impact.axis] ?? [];
    if (fullValues.length === 0) continue;

    let focusedValues: (string | number | boolean)[];

    if (fullValues.every((v) => typeof v === "number")) {
      // Numeric: focus around the best value
      focusedValues = generateFocusedNumericValues(
        fullValues as number[],
        impact.bestValue as number,
        maxPerAxis,
      );
    } else {
      // Boolean/string: use all
      focusedValues = fullValues;
    }

    for (const value of focusedValues) {
      const config = { ...winner, [impact.axis]: value };
      const key = JSON.stringify(config);
      if (!deduped.has(key)) {
        deduped.set(key, {
          variedParam: impact.axis,
          config,
        });
      }
    }
  }

  return Array.from(deduped.values());
}

/**
 * For a numeric axis, pick values densely around the best value from the
 * full set of available values.
 */
function generateFocusedNumericValues(
  allValues: number[],
  bestValue: number,
  maxCount: number,
): number[] {
  const sorted = [...allValues].sort((a, b) => a - b);
  const bestIdx = sorted.indexOf(bestValue);
  if (bestIdx === -1) {
    return coarsenValues(sorted, maxCount) as number[];
  }

  // Take a window of `maxCount` values centered on bestIdx
  const halfWindow = Math.floor(maxCount / 2);
  let startIdx = Math.max(0, bestIdx - halfWindow);
  let endIdx = startIdx + maxCount;
  if (endIdx > sorted.length) {
    endIdx = sorted.length;
    startIdx = Math.max(0, endIdx - maxCount);
  }
  return sorted.slice(startIdx, endIdx);
}

function medianOf(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
  }
  return sorted[mid] ?? null;
}
