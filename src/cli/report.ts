import { readdir, writeFile } from "fs/promises";
import path from "path";
import type { Command } from "commander";
import {
  loadRunRecordsWithDiagnostics,
  summarizeRecords,
} from "../reporter/results.js";
import {
  buildModelSummaryCsv,
  buildComparisonCsv,
} from "../reporter/csv.js";
import type { ComparisonRow } from "../reporter/csv.js";
import type { LoggerLike } from "../reporter/terminal.js";

export interface ReportDeps {
  cwd?: () => string;
  logger?: LoggerLike & { error(message: string): void };
}

export function registerReportCommand(
  command: Command,
  deps: ReportDeps,
): void {
  const cwd = deps.cwd ?? (() => process.cwd());
  const logger = deps.logger ?? console;

  command
    .command("report")
    .description("Regenerate CSV summaries from runs.jsonl files")
    .option("-i, --input <dir>", "Results directory", "results")
    .action(async (options: { input?: string }) => {
      const root = path.resolve(cwd(), options.input ?? "results");
      const entries = await readdir(root, { withFileTypes: true });
      const modelDirs = entries
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name);
      const comparisonRows: ComparisonRow[] = [];
      let hadWarnings = false;

      for (const modelDir of modelDirs) {
        const runsPath = path.join(root, modelDir, "runs.jsonl");
        try {
          const { records, invalidLineCount, hadAnyNonEmptyLine } =
            await loadRunRecordsWithDiagnostics(runsPath);
          if (
            invalidLineCount > 0 ||
            (hadAnyNonEmptyLine && records.length === 0)
          ) {
            throw new Error(
              `runs.jsonl contained ${invalidLineCount} invalid line(s)`,
            );
          }
          const summaries = summarizeRecords(records);
          if (summaries.length === 0) {
            continue;
          }
          const summaryCsv = buildModelSummaryCsv(summaries);
          await writeFile(
            path.join(root, modelDir, "summary.csv"),
            summaryCsv,
            "utf8",
          );
          const best = summaries[0];
          if (!best) {
            continue;
          }
          comparisonRows.push({
            model: modelDir,
            bestMedianTg: best.medianTg,
            bestMedianPp: best.medianPp,
            bestMedianTtft: best.medianTtft,
            bestConfigKey: best.configKey,
          });
        } catch (error) {
          hadWarnings = true;
          logger.error(
            `Skipping results for ${modelDir}: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }
      const comparisonCsv = buildComparisonCsv(comparisonRows);
      await writeFile(
        path.join(root, "comparison.csv"),
        comparisonCsv,
        "utf8",
      );
      logger.log(
        `Regenerated reports for ${comparisonRows.length} model(s).`,
      );
      if (hadWarnings) {
        process.exitCode = 1;
      }
    });
}
