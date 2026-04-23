import { Matrix, inverse } from "ml-matrix";
import fCDF from "@stdlib/stats-base-dists-f-cdf";
import type { RunRecord } from "../config/schema.js";

type Scalar = string | number | boolean;
type FactorType = "numeric" | "categorical" | "boolean";

export interface AnalyzeOptions {
  responseMetric?: "tg_tokens_per_sec" | "pp_tokens_per_sec";
  bootstrapIterations?: number;
  topInteractions?: number;
  seed?: number;
}

export interface FactorResult {
  factor: string;
  importance: number;
  share: number;
  ci: { lower: number; upper: number };
  fStatistic: number | null;
  pValue: number | null;
  direction: "positive" | "negative" | "mixed" | "unknown";
  nonlinear: boolean;
}

export interface InteractionResult {
  pair: [string, string];
  incrementalR2: number;
}

export interface AnalysisReport {
  model: {
    sampleCount: number;
    uniqueConfigs: number;
    r2: number;
    adjustedR2: number;
  };
  factors: FactorResult[];
  interactions: InteractionResult[];
}

interface Observation {
  y: number;
  config: Record<string, Scalar>;
}

interface DesignMatrix {
  x: Matrix;
  y: Matrix;
  factorColumns: Map<string, number[]>;
  factorTypes: Map<string, FactorType>;
  factorValues: Map<string, Scalar[]>;
}

interface FitResult {
  beta: number[];
  rss: number;
  tss: number;
  r2: number;
  adjustedR2: number;
  n: number;
  p: number;
}

function createRng(seed: number): () => number {
  let state = seed >>> 0 || 0x9e3779b9;
  return (): number => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.max(0, Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1))));
  return sorted[i] ?? 0;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length === 0) return 0;
  if (sorted.length % 2 === 0) {
    return (((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2);
  }
  return sorted[mid] ?? 0;
}

function toAppliedConfig(record: RunRecord): Record<string, Scalar> | undefined {
  return record.applied_config ?? record.config;
}

function selectCompleteSuccessRecords(records: RunRecord[]): Observation[] {
  const successes = records.filter((r) => r.status === "success");
  const maxKeys = successes.reduce((acc, r) => {
    const cfg = toAppliedConfig(r);
    return Math.max(acc, cfg ? Object.keys(cfg).length : 0);
  }, 0);

  const observations: Observation[] = [];
  for (const record of successes) {
    const cfg = toAppliedConfig(record);
    const tg = record.metrics.tg_tokens_per_sec;
    if (!cfg || Object.keys(cfg).length !== maxKeys || typeof tg !== "number") continue;
    observations.push({ y: tg, config: cfg });
  }
  return observations;
}

function classifyFactor(values: Scalar[]): FactorType {
  if (values.every((v) => typeof v === "boolean")) return "boolean";
  if (values.every((v) => typeof v === "number")) return "numeric";
  return "categorical";
}

function detectNonlinearity(values: Scalar[], observations: Observation[], factor: string): boolean {
  if (!values.every((v) => typeof v === "number") || values.length < 3) return false;
  const rows = values
    .map((v) => {
      const y = observations
        .filter((o) => o.config[factor] === v)
        .map((o) => o.y);
      return { value: v as number, mean: y.length > 0 ? median(y) : 0 };
    })
    .sort((a, b) => a.value - b.value);

  let previousSign = 0;
  let signChanges = 0;
  for (let i = 1; i < rows.length; i += 1) {
    const delta = (rows[i]?.mean ?? 0) - (rows[i - 1]?.mean ?? 0);
    const sign = Math.abs(delta) < 1e-6 ? 0 : delta > 0 ? 1 : -1;
    if (sign !== 0 && previousSign !== 0 && sign !== previousSign) {
      signChanges += 1;
    }
    if (sign !== 0) previousSign = sign;
  }
  return signChanges > 0;
}

function buildDesignMatrix(observations: Observation[]): DesignMatrix {
  if (observations.length === 0) {
    throw new Error("No successful complete observations found");
  }

  const factors = Object.keys(observations[0]?.config ?? {}).sort();
  const factorValues = new Map<string, Scalar[]>();
  const factorTypes = new Map<string, FactorType>();
  for (const factor of factors) {
    const unique = Array.from(
      new Set(observations.map((o) => o.config[factor] as Scalar)),
    );
    factorValues.set(factor, unique);
    factorTypes.set(factor, classifyFactor(unique));
  }

  const rows: number[][] = [];
  const factorColumns = new Map<string, number[]>();
  const interceptColumn = 0;
  let nextCol = interceptColumn + 1;

  for (const factor of factors) {
    const values = factorValues.get(factor) ?? [];
    const kind = factorTypes.get(factor) ?? "categorical";
    const cols: number[] = [];
    if (kind === "numeric") {
      cols.push(nextCol);
      nextCol += 1;
      if (values.length >= 3) {
        cols.push(nextCol);
        nextCol += 1;
      }
    } else {
      // K-1 encoding
      const width = Math.max(1, values.length - 1);
      for (let i = 0; i < width; i += 1) {
        cols.push(nextCol);
        nextCol += 1;
      }
    }
    factorColumns.set(factor, cols);
  }

  for (const obs of observations) {
    const row = new Array(nextCol).fill(0);
    row[interceptColumn] = 1;

    for (const factor of factors) {
      const kind = factorTypes.get(factor) ?? "categorical";
      const cols = factorColumns.get(factor) ?? [];
      const value = obs.config[factor];
      if (kind === "numeric") {
        const all = (factorValues.get(factor) ?? []).filter(
          (v): v is number => typeof v === "number",
        );
        const mean = all.length > 0 ? all.reduce((a, b) => a + b, 0) / all.length : 0;
        const centered = (typeof value === "number" ? value : 0) - mean;
        if (cols[0] !== undefined) row[cols[0]] = centered;
        if (cols[1] !== undefined) row[cols[1]] = centered * centered;
      } else {
        const levels = factorValues.get(factor) ?? [];
        const baseline = levels[0];
        if (value === baseline) {
          // baseline is all zeros
        } else {
          const levelIndex = levels.findIndex((v) => v === value);
          const colIndex = levelIndex - 1;
          const col = cols[colIndex];
          if (col !== undefined) row[col] = 1;
        }
      }
    }
    rows.push(row);
  }

  const x = new Matrix(rows);
  const y = Matrix.columnVector(observations.map((o) => o.y));
  return { x, y, factorColumns, factorTypes, factorValues };
}

function fitOls(x: Matrix, y: Matrix): FitResult {
  const n = x.rows;
  const p = x.columns;
  const xt = x.transpose();
  const xtx = xt.mmul(x);
  const lambda = 1e-8;
  for (let i = 0; i < xtx.rows; i += 1) {
    xtx.set(i, i, xtx.get(i, i) + lambda);
  }
  const beta = inverse(xtx).mmul(xt).mmul(y);
  const yHat = x.mmul(beta);
  const yMean = y.mean();
  let rss = 0;
  let tss = 0;
  for (let i = 0; i < y.rows; i += 1) {
    const yi = y.get(i, 0);
    const fi = yHat.get(i, 0);
    rss += (yi - fi) ** 2;
    tss += (yi - yMean) ** 2;
  }
  const r2 = tss > 0 ? 1 - rss / tss : 0;
  const adjustedR2 =
    n > p + 1 ? 1 - (1 - r2) * ((n - 1) / Math.max(1, n - p)) : r2;
  return {
    beta: beta.to1DArray(),
    rss,
    tss,
    r2,
    adjustedR2,
    n,
    p,
  };
}

function removeColumns(matrix: Matrix, toRemove: Set<number>): Matrix {
  const keep: number[] = [];
  for (let c = 0; c < matrix.columns; c += 1) {
    if (!toRemove.has(c)) keep.push(c);
  }
  const out = Matrix.zeros(matrix.rows, keep.length);
  for (let r = 0; r < matrix.rows; r += 1) {
    for (let c = 0; c < keep.length; c += 1) {
      out.set(r, c, matrix.get(r, keep[c] ?? 0));
    }
  }
  return out;
}

function uniqueConfigCount(observations: Observation[]): number {
  return new Set(observations.map((o) => JSON.stringify(o.config))).size;
}

export function analyzeFactorImportance(
  records: RunRecord[],
  options: AnalyzeOptions = {},
): AnalysisReport {
  const bootstrapIterations = options.bootstrapIterations ?? 200;
  const topInteractions = options.topInteractions ?? 3;
  const seed = options.seed ?? 42;

  const observations = selectCompleteSuccessRecords(records);
  const design = buildDesignMatrix(observations);
  const full = fitOls(design.x, design.y);

  const rawImportance = new Map<string, number>();
  const fStats = new Map<string, { f: number | null; p: number | null }>();
  const directions = new Map<string, FactorResult["direction"]>();
  const nonlinear = new Map<string, boolean>();

  for (const [factor, cols] of design.factorColumns.entries()) {
    const reducedX = removeColumns(design.x, new Set(cols));
    const reducedFit = fitOls(reducedX, design.y);
    const delta = full.tss > 0 ? Math.max(0, reducedFit.rss - full.rss) / full.tss : 0;
    rawImportance.set(factor, delta);

    const df1 = full.p - reducedFit.p;
    const df2 = Math.max(1, full.n - full.p);
    const rssDiff = Math.max(0, reducedFit.rss - full.rss);
    let fValue: number | null = null;
    let pValue: number | null = null;
    if (df1 > 0 && full.rss > 0) {
      fValue = (rssDiff / df1) / (full.rss / df2);
      if (Number.isFinite(fValue) && fValue >= 0) {
        pValue = Math.max(0, Math.min(1, 1 - fCDF(fValue, df1, df2)));
      }
    }
    fStats.set(factor, { f: fValue, p: pValue });

    const kind = design.factorTypes.get(factor) ?? "categorical";
    if (kind === "numeric") {
      const mainCoef = full.beta[cols[0] ?? -1] ?? 0;
      directions.set(
        factor,
        mainCoef > 0 ? "positive" : mainCoef < 0 ? "negative" : "unknown",
      );
    } else {
      const means = (design.factorValues.get(factor) ?? []).map((value) => {
        const ys = observations
          .filter((o) => o.config[factor] === value)
          .map((o) => o.y);
        return ys.length > 0 ? median(ys) : 0;
      });
      const min = Math.min(...means);
      const max = Math.max(...means);
      directions.set(
        factor,
        Math.abs(max - min) < 1e-9 ? "unknown" : "mixed",
      );
    }
    nonlinear.set(
      factor,
      detectNonlinearity(design.factorValues.get(factor) ?? [], observations, factor),
    );
  }

  // Bootstrap confidence intervals over grouped importances.
  const bootstrapSamples = new Map<string, number[]>();
  for (const factor of design.factorColumns.keys()) bootstrapSamples.set(factor, []);
  const rand = createRng(seed);
  for (let i = 0; i < bootstrapIterations; i += 1) {
    const sampleObs: Observation[] = [];
    for (let r = 0; r < observations.length; r += 1) {
      const idx = Math.floor(rand() * observations.length);
      const obs = observations[idx];
      if (obs) sampleObs.push(obs);
    }
    try {
      const d = buildDesignMatrix(sampleObs);
      const f = fitOls(d.x, d.y);
      for (const [factor, cols] of d.factorColumns.entries()) {
        const reduced = fitOls(removeColumns(d.x, new Set(cols)), d.y);
        const delta = f.tss > 0 ? Math.max(0, reduced.rss - f.rss) / f.tss : 0;
        bootstrapSamples.get(factor)?.push(delta);
      }
    } catch {
      // Skip degenerate bootstrap draws
    }
  }

  const totalImportance = Array.from(rawImportance.values()).reduce((a, b) => a + b, 0);
  const factors: FactorResult[] = Array.from(rawImportance.entries())
    .map(([factor, importance]) => {
      const draws = bootstrapSamples.get(factor) ?? [];
      const sorted = [...draws].sort((a, b) => a - b);
      const ciLower = draws.length > 0 ? percentile(sorted, 0.025) : importance;
      const ciUpper = draws.length > 0 ? percentile(sorted, 0.975) : importance;
      const stats = fStats.get(factor) ?? { f: null, p: null };
      return {
        factor,
        importance,
        share: totalImportance > 0 ? importance / totalImportance : 0,
        ci: { lower: ciLower, upper: ciUpper },
        fStatistic: stats.f,
        pValue: stats.p,
        direction: directions.get(factor) ?? "unknown",
        nonlinear: nonlinear.get(factor) ?? false,
      };
    })
    .sort((a, b) => b.importance - a.importance);

  // Lightweight interaction screen: pairwise product of numeric-encoded factors.
  const top = factors.slice(0, Math.max(2, topInteractions)).map((f) => f.factor);
  const interactions: InteractionResult[] = [];
  for (let i = 0; i < top.length; i += 1) {
    for (let j = i + 1; j < top.length; j += 1) {
      const a = top[i];
      const b = top[j];
      if (!a || !b) continue;
      const aVals = design.factorValues.get(a) ?? [];
      const bVals = design.factorValues.get(b) ?? [];
      const encodeA = new Map(aVals.map((v, idx) => [v, idx]));
      const encodeB = new Map(bVals.map((v, idx) => [v, idx]));
      const interCol = observations.map((o) => {
        const va = encodeA.get(o.config[a]) ?? 0;
        const vb = encodeB.get(o.config[b]) ?? 0;
        return va * vb;
      });
      const z = Matrix.zeros(design.x.rows, design.x.columns + 1);
      z.setSubMatrix(design.x, 0, 0);
      for (let r = 0; r < interCol.length; r += 1) {
        z.set(r, design.x.columns, interCol[r] ?? 0);
      }
      const withInter = fitOls(z, design.y);
      interactions.push({
        pair: [a, b],
        incrementalR2: Math.max(0, withInter.r2 - full.r2),
      });
    }
  }
  interactions.sort((l, r) => r.incrementalR2 - l.incrementalR2);

  return {
    model: {
      sampleCount: observations.length,
      uniqueConfigs: uniqueConfigCount(observations),
      r2: full.r2,
      adjustedR2: full.adjustedR2,
    },
    factors,
    interactions: interactions.slice(0, topInteractions),
  };
}
