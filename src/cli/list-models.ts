import type { Command } from "commander";
import { fileExists, readJson } from "../config/loader.js";
import { LocalStateSchema } from "../config/schema.js";
import { resolveDataRoot, statePath } from "../config/data-root.js";
import type { LoggerLike } from "../reporter/terminal.js";

export interface ListModelsDeps {
  dataRoot?: string;
  logger?: LoggerLike;
}

export function registerListModelsCommand(
  command: Command,
  deps: ListModelsDeps,
): void {
  const logger = deps.logger ?? console;

  command
    .command("list-models")
    .description("List models discovered by setup")
    .option(
      "--data-dir <path>",
      "Data directory (default: ~/.lmstudio-bench)",
    )
    .action(async (options: { dataDir?: string }) => {
      const root = resolveDataRoot(options.dataDir ?? deps.dataRoot);
      const sp = statePath(root);
      if (!(await fileExists(sp))) {
        throw new Error("Run `lmstudio-bench setup` first");
      }
      const state = await readJson(sp, LocalStateSchema);
      if (state.models.length === 0) {
        logger.log("No models discovered.");
        return;
      }
      for (const model of state.models) {
        logger.log(`${model.name}\t${model.path}`);
      }
    });
}
