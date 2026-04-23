import { writeFile } from "fs/promises";
import path from "path";
import Table from "cli-table3";
import type { Command } from "commander";
import { loadRunRecords } from "../reporter/results.js";
import { analyzeFactorImportance } from "../analysis/factor-importance.js";
import { resolveDataRoot, modelRunsPath, resultsDir } from "../config/data-root.js";
import type { LoggerLike } from "../reporter/terminal.js";

export interface AnalyzeDeps {
  dataRoot?: string;
  logger?: LoggerLike;
}

function sanitizeFilePart(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "-");
}

function modelDirName(modelRef: string): string {
  if (modelRef.includes("/") || modelRef.endsWith(".gguf")) {
    return sanitizeFilePart(path.basename(modelRef, path.extname(modelRef)));
  }
  return sanitizeFilePart(modelRef);
}

export function registerAnalyzeCommand(command: Command, deps: AnalyzeDeps): void {
  const logger = deps.logger ?? console;

  command
    .command("analyze")
    .description("Estimate statistically important factors for throughput")
    .requiredOption("-m, --model <model>", "Model name or path")
    .option("--data-dir <path>", "Data directory (default: ~/.lmstudio-bench)")
    .option("--bootstrap <n>", "Bootstrap iterations", "200")
    .option("--top-interactions <n>", "Top pair interactions to report", "3")
    .option("--json", "Write JSON report next to model runs")
    .action(
      async (options: {
        model: string;
        dataDir?: string;
        bootstrap?: string;
        topInteractions?: string;
        json?: boolean;
      }) => {
        const root = resolveDataRoot(options.dataDir ?? deps.dataRoot);
        const modelDir = modelDirName(options.model);
        const runsPath = modelRunsPath(root, modelDir);
        const records = await loadRunRecords(runsPath);
        if (records.length === 0) {
          throw new Error(`No runs found for model '${options.model}' at ${runsPath}`);
        }

        const bootstrap = Math.max(
          20,
          Number.parseInt(options.bootstrap ?? "200", 10) || 200,
        );
        const topInteractions = Math.max(
          1,
          Number.parseInt(options.topInteractions ?? "3", 10) || 3,
        );

        const report = analyzeFactorImportance(records, {
          responseMetric: "tg_tokens_per_sec",
          bootstrapIterations: bootstrap,
          topInteractions,
        });

        logger.log(
          `Model fit: n=${report.model.sampleCount}, unique=${report.model.uniqueConfigs}, r2=${report.model.r2.toFixed(4)}, adj_r2=${report.model.adjustedR2.toFixed(4)}`,
        );

        const table = new Table({
          head: [
            "factor",
            "importance",
            "share",
            "95% CI",
            "F",
            "p",
            "direction",
            "nonlinear",
          ],
        });
        for (const factor of report.factors) {
          table.push([
            factor.factor,
            factor.importance.toFixed(5),
            `${(factor.share * 100).toFixed(1)}%`,
            `[${factor.ci.lower.toFixed(5)}, ${factor.ci.upper.toFixed(5)}]`,
            factor.fStatistic !== null ? factor.fStatistic.toFixed(3) : "-",
            factor.pValue !== null ? factor.pValue.toExponential(2) : "-",
            factor.direction,
            factor.nonlinear ? "yes" : "no",
          ]);
        }
        logger.log(table.toString());

        if (report.interactions.length > 0) {
          logger.log("Top interactions:");
          for (const item of report.interactions) {
            logger.log(
              `  ${item.pair[0]} × ${item.pair[1]}: +${item.incrementalR2.toFixed(5)} r2`,
            );
          }
        }

        if (options.json) {
          const outputPath = path.join(resultsDir(root), modelDir, "analysis.json");
          await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
          logger.log(`Wrote ${outputPath}`);
        }
      },
    );
}
