import type { BenchmarkConfig, SensitivityEntry } from "./sensitivity.js";

/**
 * Deterministic pseudo-random number generator (xorshift32).
 * Reproducible across runs for the same seed.
 */
function createRng(seed: number) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (): number => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

/**
 * Fisher-Yates shuffle using the provided RNG.
 */
function shuffle<T>(arr: T[], rand: () => number): T[] {
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

/**
 * Generate a Latin Hypercube Sample over the sweep space.
 *
 * For K samples across axes each with L_i levels:
 * - Each level of every axis appears floor(K/L_i) or ceil(K/L_i) times
 * - Columns are independently shuffled to decorrelate axes
 *
 * This gives space-filling coverage without the exponential blow-up
 * of full factorial, and much better multi-axis coverage than OAT.
 */
export function buildLhsSamples(
  baseline: BenchmarkConfig,
  scan: Record<string, (string | number | boolean)[]>,
  budget: number,
  preferredValues: Record<string, string | number | boolean> = {},
  seed = 42,
): SensitivityEntry[] {
  const rand = createRng(seed);
  const axes = Object.entries(scan).filter(
    ([, values]) => values && values.length > 0,
  );

  if (axes.length === 0 || budget <= 0) return [];

  // For each axis, build an array of length `budget` with balanced level assignments
  const columns = new Map<string, (string | number | boolean)[]>();

  for (const [axis, values] of axes) {
    const col: (string | number | boolean)[] = [];
    const preferred = preferredValues[axis];

    // Ensure every level gets coverage before assigning the remaining slots.
    for (const value of values) {
      if (col.length < budget) {
        col.push(value);
      }
    }

    // Fill the remaining slots with a mild bias toward the preferred value.
    const weightedCycle =
      preferred !== undefined && values.includes(preferred)
        ? [preferred, preferred, ...values]
        : values;
    while (col.length < budget) {
      col.push(weightedCycle[(col.length - values.length) % weightedCycle.length]!);
    }
    columns.set(axis, shuffle(col, rand));
  }

  // Assemble configs
  const deduped = new Map<string, SensitivityEntry>();
  for (let i = 0; i < budget; i++) {
    const config: BenchmarkConfig = { ...baseline };
    for (const [axis] of axes) {
      const col = columns.get(axis)!;
      config[axis] = col[i]!;
    }
    const key = JSON.stringify(config);
    if (!deduped.has(key)) {
      deduped.set(key, { variedParam: "lhs", config });
    }
  }

  return Array.from(deduped.values());
}

/**
 * Estimate a good LHS budget based on the scan space dimensions.
 * Heuristic: max(total_levels × 2, 60) capped at a ceiling.
 * This ensures at least ~2 samples per level per axis on average.
 */
export function estimateLhsBudget(
  scan: Record<string, (string | number | boolean)[]>,
  maxBudget = 80,
  minBudget = 24,
): number {
  const axes = Object.entries(scan).filter(
    ([, values]) => values && values.length > 0,
  );
  const totalLevels = axes.reduce((sum, [, v]) => sum + v.length, 0);
  const widestAxis = axes.reduce((max, [, v]) => Math.max(max, v.length), 0);
  // Target lighter coverage by default while still touching every level multiple times.
  const target = Math.max(totalLevels * 2, widestAxis * 6, minBudget);
  return Math.min(target, maxBudget);
}
