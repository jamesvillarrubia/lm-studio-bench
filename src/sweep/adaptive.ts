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
