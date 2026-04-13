import chalk from "chalk";
import Table from "cli-table3";
import type { ConfigSummary } from "./results.js";
import type { ComparisonRow } from "./csv.js";

export interface LoggerLike {
  log(message: string): void;
  error?(message: string): void;
}

export interface PrintScanSummaryOptions {
  modelName?: string;
}

function formatScalarForBaseline(value: unknown): string {
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return String(value);
  }
  return JSON.stringify(value);
}

export function formatOptimalSettingsLines(
  config: Record<string, unknown>,
): string {
  const keys = Object.keys(config).sort();
  if (keys.length === 0) {
    return "  (no applied fields in summary)";
  }
  return keys
    .map((key) => {
      const value = config[key];
      if (value === undefined) {
        return `  ${key}: (missing)`;
      }
      return `  ${key}: ${formatScalarForBaseline(value)}`;
    })
    .join("\n");
}

export function printScanSummary(
  logger: LoggerLike,
  summaries: ConfigSummary[],
  options?: PrintScanSummaryOptions,
): void {
  if (summaries.length === 0) {
    logger.log("No scan results to summarize.");
    return;
  }
  const table = new Table({
    head: [
      "rank",
      "tg tokens/s",
      "pp tokens/s",
      "ttft ms",
      "success",
      "failed",
      "config",
    ],
  });
  summaries.forEach((summary, index) => {
    table.push([
      index + 1,
      summary.medianTg ?? "-",
      summary.medianPp ?? "-",
      summary.medianTtft ?? "-",
      summary.successCount,
      summary.failedCount,
      summary.configKey,
    ]);
  });
  logger.log(table.toString());
  const best = summaries[0];
  if (!best) {
    return;
  }
  logger.log(
    chalk.green(
      `Best throughput: rank #1 (tg=${best.medianTg ?? "n/a"} tokens/s)`,
    ),
  );
  const modelLabel = options?.modelName ? ` for ${options.modelName}` : "";
  logger.log("");
  if (best.successCount === 0) {
    logger.log(
      chalk.yellow(
        `No successful runs${modelLabel}; the block below is still rank #1 by ordering but medians are missing \u2014 fix failures and re-scan.`,
      ),
    );
  } else {
    logger.log(
      chalk.bold(
        `Optimal benchmark settings${modelLabel} (rank #1 by median tg tokens/s):`,
      ),
    );
  }
  logger.log(
    chalk.dim(
      "Paste under baseline in sweep-config (these are applied llama-bench fields for this build):",
    ),
  );
  logger.log(formatOptimalSettingsLines(best.config));
}

export function printModelComparison(
  logger: LoggerLike,
  rows: ComparisonRow[],
): void {
  logger.log("Model comparison");
  if (rows.length === 0) {
    logger.log("No model rows available.");
    return;
  }
  const table = new Table({
    head: ["model", "best tg", "best pp", "best ttft", "config"],
  });
  rows.forEach((row) => {
    table.push([
      row.model,
      row.bestMedianTg ?? "-",
      row.bestMedianPp ?? "-",
      row.bestMedianTtft ?? "-",
      row.bestConfigKey,
    ]);
  });
  logger.log(table.toString());
}
