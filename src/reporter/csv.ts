import type { ConfigSummary } from "./results.js";

export interface ComparisonRow {
  model: string;
  bestMedianTg: number | null;
  bestMedianPp: number | null;
  bestMedianTtft: number | null;
  bestConfigKey: string;
}

function escapeCsvCell(value: unknown): string {
  if (value === null) {
    return "";
  }
  const raw = String(value);
  if (raw.includes(",") || raw.includes('"') || raw.includes("\n")) {
    return `"${raw.replaceAll('"', '""')}"`;
  }
  return raw;
}

export function buildModelSummaryCsv(summaries: ConfigSummary[]): string {
  const header =
    "rank,median_tg_tokens_per_sec,median_pp_tokens_per_sec,median_ttft_ms,success_count,failed_count,config_key";
  const lines = summaries.map((summary, index) =>
    [
      index + 1,
      summary.medianTg,
      summary.medianPp,
      summary.medianTtft,
      summary.successCount,
      summary.failedCount,
      summary.configKey,
    ]
      .map((entry) => escapeCsvCell(entry))
      .join(","),
  );
  return [header, ...lines].join("\n").concat("\n");
}

export function buildComparisonCsv(rows: ComparisonRow[]): string {
  const header =
    "model,best_median_tg_tokens_per_sec,best_median_pp_tokens_per_sec,best_median_ttft_ms,best_config_key";
  const lines = rows.map((row) =>
    [
      row.model,
      row.bestMedianTg,
      row.bestMedianPp,
      row.bestMedianTtft,
      row.bestConfigKey,
    ]
      .map((entry) => escapeCsvCell(entry))
      .join(","),
  );
  return [header, ...lines].join("\n").concat("\n");
}
