import { mkdir, readdir, writeFile } from "fs/promises";
import path from "path";
import type { Command } from "commander";
import {
  resolveDataRoot,
  resultsDir,
} from "../config/data-root.js";
import {
  loadRunRecords,
  summarizeRecords,
} from "../reporter/results.js";
import type { RunRecord } from "../config/schema.js";
import type { LoggerLike } from "../reporter/terminal.js";
import { analyzeFactorImportance } from "../analysis/factor-importance.js";

type Scalar = string | number | boolean;

interface AxisValueBucket {
  value: Scalar;
  raw: number[];
  median: number;
  count: number;
}

interface AxisEffect {
  axis: string;
  type: "numeric" | "categorical";
  values: AxisValueBucket[];
  range: number;
}

interface FactorRecommendation {
  factor: string;
  value: Scalar;
  medianTg: number;
  count: number;
}

interface CoverageAxisStatus {
  axis: string;
  minCount: number;
  maxCount: number;
  levels: { value: Scalar; count: number }[];
  sparseValues: Scalar[];
  status: "good" | "sparse";
}

/** Preferred axis order for correlation matrix labels (then remaining keys A–Z). */
const CORRELATION_AXIS_PRIORITY = [
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
  "n_cpu_moe",
] as const;

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((s, x) => s + x, 0) / values.length;
}

function variance(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return values.reduce((s, x) => s + (x - m) * (x - m), 0) / (values.length - 1);
}

function pearsonCorrelation(a: number[], b: number[]): number | null {
  const n = a.length;
  if (n < 3 || n !== b.length) return null;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i += 1) {
    const x = a[i]! - ma;
    const y = b[i]! - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  if (da < 1e-14 || db < 1e-14) return null;
  const r = num / Math.sqrt(da * db);
  return Math.max(-1, Math.min(1, r));
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    const left = sorted[mid - 1] ?? 0;
    const right = sorted[mid] ?? 0;
    return Number(((left + right) / 2).toFixed(3));
  }
  return Number((sorted[mid] ?? 0).toFixed(3));
}

function sanitizeFilePart(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "-");
}

function appliedConfig(record: RunRecord): Record<string, Scalar> | undefined {
  return record.applied_config ?? record.config;
}

function selectCompleteSuccessRecords(records: RunRecord[]): RunRecord[] {
  const successes = records.filter((r) => r.status === "success");
  const maxKeys = successes.reduce((max, r) => {
    const cfg = appliedConfig(r);
    return Math.max(max, cfg ? Object.keys(cfg).length : 0);
  }, 0);
  return successes.filter((r) => {
    const cfg = appliedConfig(r);
    return cfg && Object.keys(cfg).length === maxKeys;
  });
}

function computeAxisEffects(records: RunRecord[]): AxisEffect[] {
  const byAxisValue = new Map<string, Map<string, number[]>>();
  const valueTypes = new Map<string, "numeric" | "categorical">();

  for (const record of records) {
    const cfg = appliedConfig(record);
    const tg = record.metrics.tg_tokens_per_sec;
    if (!cfg || typeof tg !== "number") continue;

    for (const [axis, rawValue] of Object.entries(cfg)) {
      let axisMap = byAxisValue.get(axis);
      if (!axisMap) {
        axisMap = new Map<string, number[]>();
        byAxisValue.set(axis, axisMap);
      }
      const valueKey = JSON.stringify(rawValue);
      const bucket = axisMap.get(valueKey) ?? [];
      bucket.push(tg);
      axisMap.set(valueKey, bucket);

      if (!valueTypes.has(axis)) {
        valueTypes.set(axis, typeof rawValue === "number" ? "numeric" : "categorical");
      }
    }
  }

  const effects: AxisEffect[] = [];
  for (const [axis, axisMap] of byAxisValue.entries()) {
    const rows: AxisValueBucket[] = Array.from(axisMap.entries()).map(([raw, tgs]) => ({
      value: JSON.parse(raw) as Scalar,
      raw: tgs,
      median: median(tgs),
      count: tgs.length,
    }));
    const type = valueTypes.get(axis) ?? "categorical";
    if (type === "numeric") {
      rows.sort((a, b) => Number(a.value) - Number(b.value));
    } else {
      rows.sort((a, b) => String(a.value).localeCompare(String(b.value)));
    }
    const medians = rows.map((r) => r.median);
    const range = medians.length > 0 ? Math.max(...medians) - Math.min(...medians) : 0;
    effects.push({
      axis,
      type,
      values: rows,
      range: Number(range.toFixed(3)),
    });
  }

  effects.sort((a, b) => b.range - a.range);
  return effects;
}

function computeGlobalMedianTg(records: RunRecord[]): number {
  const tgs = records
    .map((r) => r.metrics.tg_tokens_per_sec)
    .filter((v): v is number => typeof v === "number");
  if (tgs.length === 0) return 0;
  return median(tgs);
}

function computeFactorRecommendations(records: RunRecord[]): FactorRecommendation[] {
  const byAxisValue = new Map<string, Map<string, { value: Scalar; tg: number[] }>>();
  for (const record of records) {
    const cfg = appliedConfig(record);
    const tg = record.metrics.tg_tokens_per_sec;
    if (!cfg || typeof tg !== "number") continue;
    for (const [axis, value] of Object.entries(cfg)) {
      let axisMap = byAxisValue.get(axis);
      if (!axisMap) {
        axisMap = new Map();
        byAxisValue.set(axis, axisMap);
      }
      const key = JSON.stringify(value);
      const bucket = axisMap.get(key) ?? { value, tg: [] };
      bucket.tg.push(tg);
      axisMap.set(key, bucket);
    }
  }

  const out: FactorRecommendation[] = [];
  for (const [factor, values] of byAxisValue.entries()) {
    let best: FactorRecommendation | null = null;
    for (const bucket of values.values()) {
      const med = median(bucket.tg);
      const candidate: FactorRecommendation = {
        factor,
        value: bucket.value,
        medianTg: med,
        count: bucket.tg.length,
      };
      if (!best || candidate.medianTg > best.medianTg) {
        best = candidate;
      }
    }
    if (best) out.push(best);
  }
  return out;
}

function computeCoverageStatus(effects: AxisEffect[]): CoverageAxisStatus[] {
  return effects.map((effect) => {
    const levels = effect.values.map((v) => ({
      value: v.value,
      count: v.count,
    }));
    const counts = levels.map((l) => l.count);
    const minCount = counts.length > 0 ? Math.min(...counts) : 0;
    const maxCount = counts.length > 0 ? Math.max(...counts) : 0;
    const sparseValues = effect.values.filter((v) => v.count < 3).map((v) => v.value);
    return {
      axis: effect.axis,
      minCount,
      maxCount,
      levels,
      sparseValues,
      status: sparseValues.length > 0 ? "sparse" : "good",
    };
  });
}

/** Order axes by regression importance, then any remaining axes by empirical range. */
function orderEffectsForDisplay(
  effects: AxisEffect[],
  rankedFactors: { factor: string }[],
): AxisEffect[] {
  const byAxis = new Map(effects.map((e) => [e.axis, e]));
  const ordered: AxisEffect[] = [];
  const seen = new Set<string>();
  for (const { factor } of rankedFactors) {
    const e = byAxis.get(factor);
    if (e) {
      ordered.push(e);
      seen.add(factor);
    }
  }
  const rest = effects.filter((e) => !seen.has(e.axis)).sort((a, b) => b.range - a.range);
  ordered.push(...rest);
  return ordered;
}

function yamlScalar(value: Scalar): string {
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  return JSON.stringify(String(value));
}

function buildRecommendedYamlBlock(args: {
  bestConfig: Record<string, Scalar>;
  rankedFactors: { factor: string; share: number }[];
}): string {
  const shareByFactor = new Map(
    args.rankedFactors.map((f) => [f.factor, f.share]),
  );
  const keys = Object.keys(args.bestConfig).sort((a, b) => a.localeCompare(b));
  const rows: { left: string; comment: string | null }[] = [];
  for (const key of keys) {
    const v = args.bestConfig[key];
    if (v === undefined) continue;
    const left = `${key}: ${yamlScalar(v)}`;
    const share = shareByFactor.get(key);
    const comment =
      share !== undefined
        ? `# ${(share * 100).toFixed(1)}% model importance share`
        : null;
    rows.push({ left, comment });
  }
  const maxLeftLen = rows.reduce((max, r) => {
    if (r.comment === null) return max;
    return Math.max(max, r.left.length);
  }, 0);
  /** Column where `#` starts (0-based); only rows with comments are padded. */
  const commentColumn = maxLeftLen + 2;
  const lines = rows.map(({ left, comment }) => {
    if (comment === null) return left;
    const pad = Math.max(2, commentColumn - left.length);
    return `${left}${" ".repeat(pad)}${comment}`;
  });
  return lines.join("\n");
}

function buildInteractionHeatmap(
  records: RunRecord[],
  axisA: string,
  axisB: string,
): { x: string[]; y: string[]; z: (number | null)[][]; n: number[][] } {
  const aValues = new Set<string>();
  const bValues = new Set<string>();
  const buckets = new Map<string, number[]>();

  for (const record of records) {
    const cfg = appliedConfig(record);
    const tg = record.metrics.tg_tokens_per_sec;
    if (!cfg || typeof tg !== "number") continue;
    if (!(axisA in cfg) || !(axisB in cfg)) continue;

    const a = String(cfg[axisA]);
    const b = String(cfg[axisB]);
    aValues.add(a);
    bValues.add(b);
    const key = `${a}:::${b}`;
    const list = buckets.get(key) ?? [];
    list.push(tg);
    buckets.set(key, list);
  }

  const x = Array.from(aValues).sort((l, r) => l.localeCompare(r, undefined, { numeric: true }));
  const y = Array.from(bValues).sort((l, r) => l.localeCompare(r, undefined, { numeric: true }));
  const z: (number | null)[][] = [];
  const n: number[][] = [];
  for (const b of y) {
    const zRow: (number | null)[] = [];
    const nRow: number[] = [];
    for (const a of x) {
      const list = buckets.get(`${a}:::${b}`);
      const len = list?.length ?? 0;
      nRow.push(len);
      if (!list || len === 0) {
        zRow.push(null);
      } else {
        zRow.push(median(list));
      }
    }
    z.push(zRow);
    n.push(nRow);
  }
  return { x, y, z, n };
}

function sortAxisNamesForCorrelation(names: string[]): string[] {
  const priority = new Map<string, number>();
  for (const [i, axis] of CORRELATION_AXIS_PRIORITY.entries()) {
    priority.set(axis, i);
  }
  return [...names].sort((a, b) => {
    const pa = priority.get(a);
    const pb = priority.get(b);
    if (pa !== undefined && pb !== undefined) return pa - pb;
    if (pa !== undefined) return -1;
    if (pb !== undefined) return 1;
    return a.localeCompare(b);
  });
}

/**
 * Pearson correlation between sweep parameters across runs (design matrix).
 * Booleans → 0/1; strings with ≤24 levels → ordinal codes (sorted); constants excluded.
 */
function computeDesignCorrelationMatrix(records: RunRecord[]): {
  labels: string[];
  z: number[][];
  encodingByAxis: Record<string, string>;
  highCorrelations: { a: string; b: string; r: number }[];
} | null {
  if (records.length < 3) return null;
  const first = appliedConfig(records[0]);
  if (!first) return null;
  const keys = sortAxisNamesForCorrelation(Object.keys(first));

  const columns: { name: string; values: number[]; encoding: string }[] = [];

  for (const key of keys) {
    const raw: Scalar[] = [];
    for (const r of records) {
      const c = appliedConfig(r);
      if (!c || !(key in c)) {
        raw.length = 0;
        break;
      }
      raw.push(c[key]!);
    }
    if (raw.length !== records.length) continue;

    const allNumber = raw.every((x) => typeof x === "number");
    const allBool = raw.every((x) => typeof x === "boolean");
    const allString = raw.every((x) => typeof x === "string");

    if (allNumber) {
      const values = raw.map((x) => Number(x));
      if (variance(values) < 1e-12) continue;
      columns.push({ name: key, values, encoding: "numeric" });
      continue;
    }
    if (allBool) {
      const values = raw.map((x) => (x ? 1 : 0));
      if (variance(values) < 1e-12) continue;
      columns.push({ name: key, values, encoding: "boolean (0/1)" });
      continue;
    }
    if (allString) {
      const uniq = [...new Set(raw.map((x) => String(x)))].sort();
      if (uniq.length < 2 || uniq.length > 24) continue;
      const rank = new Map(uniq.map((s, i) => [s, i]));
      const values = raw.map((x) => rank.get(String(x))!);
      if (variance(values) < 1e-12) continue;
      columns.push({
        name: key,
        values,
        encoding: `ordinal (${uniq.length} levels, A→Z order)`,
      });
    }
  }

  if (columns.length < 2) return null;

  const labels = columns.map((c) => c.name);
  const encodingByAxis: Record<string, string> = {};
  for (const c of columns) {
    encodingByAxis[c.name] = c.encoding;
  }

  const m = columns.length;
  const z: number[][] = Array.from({ length: m }, () => Array.from({ length: m }, () => 0));
  for (let i = 0; i < m; i += 1) {
    z[i]![i] = 1;
    for (let j = i + 1; j < m; j += 1) {
      const r = pearsonCorrelation(columns[i]!.values, columns[j]!.values) ?? 0;
      z[i]![j] = r;
      z[j]![i] = r;
    }
  }

  const highCorrelations: { a: string; b: string; r: number }[] = [];
  for (let i = 0; i < m; i += 1) {
    for (let j = i + 1; j < m; j += 1) {
      const r = z[i]![j]!;
      if (Math.abs(r) >= 0.7) {
        highCorrelations.push({
          a: labels[i]!,
          b: labels[j]!,
          r,
        });
      }
    }
  }
  highCorrelations.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));

  return { labels, z, encodingByAxis, highCorrelations };
}

function buildHtmlReport(args: {
  model: string;
  completeCount: number;
  totalSuccess: number;
  effectsForBoxes: AxisEffect[];
  globalMedianTg: number;
  bestConfig: Record<string, Scalar>;
  bestMedianTg: number | null;
  rankedFactors: {
    factor: string;
    importance: number;
    share: number;
    ci: { lower: number; upper: number };
    nonlinear: boolean;
    direction: string;
  }[];
  recommendations: FactorRecommendation[];
  modelFit: {
    r2: number;
    adjustedR2: number;
    sampleCount: number;
    uniqueConfigs: number;
  };
  coverage: CoverageAxisStatus[];
  recommendedYaml: string;
  correlation: {
    labels: string[];
    z: number[][];
    encodingByAxis: Record<string, string>;
    highCorrelations: { a: string; b: string; r: number }[];
  } | null;
  interaction?: {
    axisA: string;
    axisB: string;
    x: string[];
    y: string[];
    z: (number | null)[][];
    n: number[][];
  };
}): string {
  const payload = JSON.stringify(args);
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>lmstudio-bench visualization</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; max-width: 1200px; }
    .chart { width: 100%; height: 360px; margin-bottom: 26px; border: 1px solid #ddd; border-radius: 6px; }
    .subtle { color: #666; font-size: 13px; }
    .banner {
      display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
      padding: 14px 18px; margin-bottom: 20px;
      background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
      border: 1px solid #7dd3fc; border-radius: 8px;
    }
    .banner strong { font-size: 1.1rem; }
    .badge-high { background: #16a34a; color: #fff; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .badge-med { background: #ca8a04; color: #fff; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .badge-low { background: #dc2626; color: #fff; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .config-block {
      position: relative; background: #1e293b; color: #e2e8f0; padding: 16px 16px 16px 16px;
      border-radius: 8px; font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.5;
      overflow-x: auto; margin-bottom: 8px;
    }
    .config-block .muted { color: #64748b; }
    button.copy-btn {
      margin-bottom: 20px; padding: 8px 16px; font-size: 14px; cursor: pointer;
      background: #2563eb; color: #fff; border: none; border-radius: 6px; font-weight: 600;
    }
    button.copy-btn:hover { background: #1d4ed8; }
    .alert {
      padding: 10px 14px; margin-bottom: 10px; border-radius: 6px;
      background: #fef3c7; border: 1px solid #f59e0b; color: #78350f; font-size: 14px;
    }
    h2 { margin-top: 32px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
  </style>
</head>
<body>
  <h1>lmstudio-bench: actionable tuning report</h1>
  <div id="banner" class="banner"></div>
  <h2>Recommended baseline (copy into sweep-config)</h2>
  <button type="button" class="copy-btn" id="copy-yaml">Copy YAML block</button>
  <pre id="yaml-block" class="config-block"></pre>
  <p class="subtle" id="yaml-footnote"></p>
  <h2>Statistical factor importance</h2>
  <p class="subtle">Bar color reflects CI tightness vs. importance (green = more trustworthy; orange = noisy / uncertain estimate).</p>
  <div id="importance" class="chart"></div>
  <h2>Design correlation matrix (Pearson)</h2>
  <p class="subtle">Linear association between sweep parameters across the same runs (not tg). Booleans are 0/1; string factors use ordinal codes in A–Z level order (treat as exploratory only). Strong |r| flags confounded exploration or coupled settings — interpret main effects and regression with care.</p>
  <p class="subtle" id="cor-encoding"></p>
  <div id="correlation" class="chart" style="height: 520px;"></div>
  <div id="cor-highpairs"></div>
  <h2>Main effects (distribution + sample size)</h2>
  <p class="subtle">Box plots use raw per-run tg tok/s. Dashed line = global median across all runs. ★ = best observed level for that axis. Prefer levels with tight boxes and n ≥ 3.</p>
  <div id="effects"></div>
  <div id="coverage-section"></div>
  <h2>Top interaction (median tg, cell count)</h2>
  <p class="subtle">Gray cells = no observations (coverage gap). Text shows n per cell.</p>
  <div id="interaction" class="chart" style="height: 420px;"></div>
  <script>
    const data = ${payload};

    function confidenceLabel(r2) {
      if (r2 >= 0.7) return { text: "HIGH", cls: "badge-high" };
      if (r2 >= 0.5) return { text: "MEDIUM", cls: "badge-med" };
      return { text: "LOW", cls: "badge-low" };
    }

    const conf = confidenceLabel(data.modelFit.r2);
    const bestTg = data.bestMedianTg != null ? data.bestMedianTg.toFixed(2) : "—";
    document.getElementById("banner").innerHTML = \`
      <span><strong>Best observed median tg:</strong> \${bestTg} tok/s</span>
      <span><span class="\${conf.cls}">Model confidence: \${conf.text}</span></span>
      <span class="subtle" style="margin:0">R² = \${data.modelFit.r2.toFixed(3)} · n = \${data.modelFit.sampleCount} runs · \${data.modelFit.uniqueConfigs} configs</span>
      <span class="subtle" style="margin:0">Complete runs in this report: \${data.completeCount} · All successes: \${data.totalSuccess}</span>
    \`;

    const yamlText = data.recommendedYaml;
    document.getElementById("yaml-block").textContent = yamlText;
    document.getElementById("yaml-footnote").textContent =
      "Paste under baseline: in sweep-config.yaml, then run confirmatory scan to validate.";

    document.getElementById("copy-yaml").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(yamlText);
        const b = document.getElementById("copy-yaml");
        const prev = b.textContent;
        b.textContent = "Copied!";
        setTimeout(() => { b.textContent = prev; }, 2000);
      } catch (e) {
        alert("Copy failed — select the block manually.");
      }
    });

    const topRev = data.rankedFactors.slice(0, 12).slice().reverse();
    const barColors = topRev.map(f => {
      const width = Math.max(1e-9, f.ci.upper - f.ci.lower);
      const rel = width / Math.max(1e-9, f.importance);
      const t = Math.min(1, rel / 2.5);
      const r = Math.round(34 + (234 - 34) * t);
      const g = Math.round(197 + (88 - 197) * t);
      const b = Math.round(94 + (34 - 94) * t);
      return \`rgb(\${r},\${g},\${b})\`;
    });
    const hoverText = topRev.map(f =>
      f.factor +
        "\\nimportance: " + f.importance.toFixed(5) +
        "\\nshare: " + (f.share * 100).toFixed(1) + "%" +
        "\\n95% CI: [" + f.ci.lower.toFixed(5) + ", " + f.ci.upper.toFixed(5) + "]" +
        "\\ndirection: " + f.direction +
        (f.nonlinear ? "\\nnonlinear: yes" : "")
    );

    Plotly.newPlot("importance", [{
      type: "bar",
      orientation: "h",
      y: topRev.map(f => f.factor),
      x: topRev.map(f => f.importance),
      error_x: {
        type: "data",
        array: topRev.map(f => Math.max(0, f.ci.upper - f.importance)),
        arrayminus: topRev.map(f => Math.max(0, f.importance - f.ci.lower)),
        visible: true
      },
      marker: { color: barColors },
      text: topRev.map(f => \`\${(f.share * 100).toFixed(1)}%\`),
      textposition: "outside",
      hovertext: hoverText,
      hoverinfo: "text+x"
    }], {
      margin: { t: 20, l: 150, r: 50, b: 50 },
      xaxis: { title: "Importance (ΔRSS / TSS)" },
    });

    if (data.correlation && data.correlation.labels.length >= 2) {
      const enc = data.correlation.encodingByAxis;
      document.getElementById("cor-encoding").textContent =
        "Encoding: " + data.correlation.labels.map((k) => k + " → " + enc[k]).join(" · ");
      const text = data.correlation.z.map((row) =>
        row.map((v) => v.toFixed(2)),
      );
      Plotly.newPlot("correlation", [{
        type: "heatmap",
        x: data.correlation.labels,
        y: data.correlation.labels,
        z: data.correlation.z,
        text,
        texttemplate: "%{text}",
        textfont: { size: 10 },
        colorscale: "RdBu",
        zmid: 0,
        zmin: -1,
        zmax: 1,
        reversescale: true,
        hovertemplate: "%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar: { title: "Pearson r" },
      }], {
        margin: { t: 30, l: 130, r: 80, b: 130 },
        xaxis: { side: "bottom", tickangle: -45 },
        yaxis: { autorange: "reversed" },
      });
      const hp = data.correlation.highCorrelations;
      const hpEl = document.getElementById("cor-highpairs");
      if (hp.length > 0) {
        hpEl.innerHTML = "<p><strong>Strong pairs (|r| ≥ 0.7)</strong></p><ul>" +
          hp.map((p) =>
            "<li><code>" + p.a + "</code> vs <code>" + p.b + "</code>: r = " + p.r.toFixed(3) + "</li>",
          ).join("") +
          "</ul><p class='subtle'>If the sweep confounds these axes, apparent factor importance can be shared between them.</p>";
      } else {
        hpEl.innerHTML = "<p class='subtle'>No off-diagonal pairs with |r| ≥ 0.7.</p>";
      }
    } else {
      document.getElementById("cor-encoding").textContent = "";
      document.getElementById("correlation").innerHTML =
        "<p class='subtle' style='padding:16px'>Not enough numeric-like parameters (need ≥ 2 non-constant columns and ≥ 3 runs).</p>";
      document.getElementById("cor-highpairs").innerHTML = "";
    }

    const effectsRoot = document.getElementById("effects");
    const shareByFactor = new Map(data.rankedFactors.map(f => [f.factor, f.share]));
    const recByFactor = new Map(data.recommendations.map(r => [r.factor, r]));

    data.effectsForBoxes.forEach((effect, idx) => {
      const title = document.createElement("h3");
      const share = shareByFactor.get(effect.axis);
      const rf = data.rankedFactors.find(f => f.factor === effect.axis);
      const sub = [
        share != null ? \`model share \${(share * 100).toFixed(1)}%\` : null,
        rf ? \`direction: \${rf.direction}\` : null,
        rf && rf.nonlinear ? "nonlinear (caution)" : null,
      ].filter(Boolean).join(" · ");
      title.textContent = effect.axis + (sub ? " — " + sub : "");
      effectsRoot.appendChild(title);
      const div = document.createElement("div");
      div.id = "eff-" + idx;
      div.className = "chart";
      effectsRoot.appendChild(div);

      const rec = recByFactor.get(effect.axis);
      const recValStr = rec ? JSON.stringify(rec.value) : "";

      const x = [];
      const y = [];
      const customdata = [];
      const minCount = Math.min(...effect.values.map(v => v.count));
      for (const v of effect.values) {
        const valStr = String(v.value);
        for (const pt of v.raw) {
          x.push(valStr);
          y.push(pt);
          customdata.push([v.count, v.median]);
        }
      }

      const boxpoints = minCount < 5 ? "all" : "suspectedoutliers";
      const jitter = minCount < 5 ? 0.25 : 0.12;

      const annotations = [];
      if (rec && recValStr) {
        const bestIdx = effect.values.findIndex(v => JSON.stringify(v.value) === recValStr);
        if (bestIdx >= 0) {
          annotations.push({
            x: String(effect.values[bestIdx].value),
            y: 1.02,
            xref: "x",
            yref: "paper",
            text: "★ best level (by median)",
            showarrow: false,
            font: { size: 11, color: "#16a34a" },
          });
        }
      }

      Plotly.newPlot(div.id, [{
        type: "box",
        x, y,
        customdata,
        marker: { color: "#3b82f6", size: 4, opacity: 0.7 },
        line: { color: "#1e40af", width: 1.5 },
        boxpoints,
        jitter,
        pointpos: 0,
        fillcolor: "rgba(59,130,246,0.12)",
        hovertemplate: "tg=%{y:.2f}<br>group n=%{customdata[0]}<br>group median=%{customdata[1]:.2f}<extra></extra>",
      }], {
        yaxis: { title: "tg tok/s (per run)" },
        xaxis: { title: effect.axis, tickangle: -35 },
        margin: { t: 36, b: 80 },
        shapes: [{
          type: "line",
          x0: 0,
          x1: 1,
          xref: "paper",
          y0: data.globalMedianTg,
          y1: data.globalMedianTg,
          yref: "y",
          line: { dash: "dash", color: "#64748b", width: 2 },
        }],
        annotations,
        showlegend: false,
      });
    });

    const covRoot = document.getElementById("coverage-section");
    const sparse = data.coverage.filter(c => c.status === "sparse");
    if (sparse.length > 0) {
      const h = document.createElement("h2");
      h.textContent = "Coverage warnings (sparse levels)";
      covRoot.appendChild(h);
      for (const c of sparse) {
        const d = document.createElement("div");
        d.className = "alert";
        const vals = c.sparseValues.map(v => String(v)).join(", ");
        d.innerHTML = "<strong>" + c.axis + "</strong> — levels with &lt; 3 runs: " + vals +
          " <span class='subtle'>(min n=" + c.minCount + ", max n=" + c.maxCount + ")</span>";
        covRoot.appendChild(d);
      }
      const cap = document.createElement("p");
      cap.className = "subtle";
      cap.textContent = "Action: add more runs for those levels, or run confirmatory OAT pinned to your best config.";
      covRoot.appendChild(cap);
      const minDiv = document.createElement("div");
      minDiv.id = "coverage-bars";
      minDiv.className = "chart";
      minDiv.style.height = "280px";
      covRoot.appendChild(minDiv);
      Plotly.newPlot("coverage-bars", [{
        type: "bar",
        x: data.coverage.map(c => c.axis),
        y: data.coverage.map(c => c.minCount),
        marker: {
          color: data.coverage.map(c => c.status === "sparse" ? "#f97316" : "#22c55e"),
        },
        text: data.coverage.map(c => "min n=" + c.minCount),
        textposition: "outside",
        hovertemplate: "%{x}<br>min runs/level: %{y}<extra></extra>",
      }], {
        yaxis: { title: "Minimum runs at any level", rangemode: "tozero" },
        xaxis: { title: "Factor", tickangle: -40 },
        shapes: [{
          type: "line",
          x0: -0.5, x1: data.coverage.length - 0.5,
          xref: "x",
          y0: 3, y1: 3,
          line: { color: "#dc2626", dash: "dash", width: 2 },
        }],
        annotations: [{
          x: data.coverage.length - 1,
          y: 3,
          text: "n=3 threshold",
          showarrow: false,
          yshift: 10,
          font: { color: "#dc2626", size: 11 },
        }],
        margin: { t: 20, b: 100 },
      });
    }

    if (data.interaction) {
      const z = data.interaction.z.map(row =>
        row.map((cell, j) => (cell === null ? null : cell))
      );
      const text = data.interaction.n.map((row, i) =>
        row.map((count, j) => {
          if (count === 0) return "no data";
          return "n=" + count + "<br>median=" + (z[i][j] != null ? z[i][j].toFixed(2) : "");
        })
      );
      let best = -Infinity;
      let bi = -1, bj = -1;
      for (let i = 0; i < z.length; i++) {
        for (let j = 0; j < (z[i] || []).length; j++) {
          const v = z[i][j];
          if (v != null && v > best) { best = v; bi = i; bj = j; }
        }
      }
      const zFin = z.flat().filter(v => v != null);
      const zmin = zFin.length ? Math.min(...zFin) : 0;
      const zmax = zFin.length ? Math.max(...zFin) : 1;

      Plotly.newPlot("interaction", [{
        type: "heatmap",
        x: data.interaction.x,
        y: data.interaction.y,
        z,
        text,
        texttemplate: "%{text}",
        textfont: { size: 10 },
        hoverongaps: false,
        colorscale: [
          [0, "#0f172a"],
          [0.35, "#0369a1"],
          [0.65, "#22c55e"],
          [1, "#fde047"],
        ],
        zmin,
        zmax,
        connectgaps: false,
        colorbar: { title: "median tg" },
      }], {
        xaxis: { title: data.interaction.axisA },
        yaxis: { title: data.interaction.axisB },
        margin: { t: 20, l: 120 },
        plot_bgcolor: "#d4d4d8",
      });

      if (bi >= 0 && bj >= 0 && best > -Infinity) {
        Plotly.relayout("interaction", {
          annotations: [{
            x: data.interaction.x[bj],
            y: data.interaction.y[bi],
            text: "★ best",
            showarrow: true,
            arrowhead: 2,
            ax: 0,
            ay: -30,
            font: { color: "#1e40af", size: 12 },
          }],
        });
      }
    } else {
      document.getElementById("interaction").textContent = "Not enough data for interaction heatmap.";
    }
  </script>
</body>
</html>`;
}

export interface VisualizeDeps {
  dataRoot?: string;
  logger?: LoggerLike;
}

export function registerVisualizeCommand(
  command: Command,
  deps: VisualizeDeps,
): void {
  const logger = deps.logger ?? console;

  command
    .command("visualize")
    .description("Generate interactive plots for variable impacts")
    .option("-m, --model <name>", "Model name (results directory)")
    .option("--data-dir <path>", "Data directory (default: ~/.lmstudio-bench)")
    .option(
      "-o, --out <dir>",
      "Output directory for HTML report (default: <data-dir>/results/<model>/plots)",
    )
    .action(async (options: { model?: string; dataDir?: string; out?: string }) => {
      const root = resolveDataRoot(options.dataDir ?? deps.dataRoot);
      const resultsRoot = resultsDir(root);

      let modelDir = options.model;
      if (!modelDir) {
        const entries = await readdir(resultsRoot, { withFileTypes: true });
        const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);
        if (dirs.length !== 1) {
          throw new Error("Provide --model when multiple result directories exist");
        }
        modelDir = dirs[0];
      }
      const runsPath = path.join(resultsRoot, sanitizeFilePart(modelDir), "runs.jsonl");
      const records = await loadRunRecords(runsPath);
      const successCount = records.filter((r) => r.status === "success").length;
      const complete = selectCompleteSuccessRecords(records);
      if (complete.length === 0) {
        throw new Error("No complete successful runs found to visualize");
      }

      const effects = computeAxisEffects(complete);
      const analysis = analyzeFactorImportance(complete, {
        bootstrapIterations: 200,
        topInteractions: 3,
      });
      const recommendations = computeFactorRecommendations(complete);

      const bestSummary = summarizeRecords(complete)[0];
      const rankedFactors = analysis.factors.map((f) => ({
        factor: f.factor,
        importance: f.importance,
        share: f.share,
        ci: f.ci,
        nonlinear: f.nonlinear,
        direction: f.direction,
      }));
      const orderedForBoxes = orderEffectsForDisplay(effects, rankedFactors).slice(0, 10);
      const coverage = computeCoverageStatus(effects);
      const recommendedYaml = buildRecommendedYamlBlock({
        bestConfig: (bestSummary?.config ?? {}) as Record<string, Scalar>,
        rankedFactors,
      });

      const topInteraction = analysis.interactions[0];
      const fallbackTopTwo = effects.slice(0, 2);
      const interaction =
        topInteraction
          ? (() => {
              const axisA = topInteraction.pair[0];
              const axisB = topInteraction.pair[1];
              if (!axisA || !axisB) return undefined;
              return { axisA, axisB, ...buildInteractionHeatmap(complete, axisA, axisB) };
            })()
          : fallbackTopTwo.length >= 2
            ? (() => {
                const axisA = fallbackTopTwo[0]?.axis;
                const axisB = fallbackTopTwo[1]?.axis;
                if (!axisA || !axisB) return undefined;
                return { axisA, axisB, ...buildInteractionHeatmap(complete, axisA, axisB) };
              })()
            : undefined;

      const correlation = computeDesignCorrelationMatrix(complete);

      const outDir =
        options.out ??
        path.join(resultsRoot, sanitizeFilePart(modelDir), "plots");
      await mkdir(outDir, { recursive: true });
      const html = buildHtmlReport({
        model: modelDir,
        completeCount: complete.length,
        totalSuccess: successCount,
        effectsForBoxes: orderedForBoxes,
        globalMedianTg: computeGlobalMedianTg(complete),
        bestConfig: (bestSummary?.config ?? {}) as Record<string, Scalar>,
        bestMedianTg: bestSummary?.medianTg ?? null,
        rankedFactors,
        recommendations,
        modelFit: {
          r2: analysis.model.r2,
          adjustedR2: analysis.model.adjustedR2,
          sampleCount: analysis.model.sampleCount,
          uniqueConfigs: analysis.model.uniqueConfigs,
        },
        coverage,
        recommendedYaml,
        correlation,
        interaction,
      });
      const outPath = path.join(outDir, "impact-report.html");
      await writeFile(outPath, html, "utf8");
      logger.log(`Visualization written: ${outPath}`);
    });
}
