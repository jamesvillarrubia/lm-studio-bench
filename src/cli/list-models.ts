import path from "path";
import type { Command } from "commander";
import { fileExists, readJson } from "../config/loader.js";
import { LocalStateSchema } from "../config/schema.js";
import type { LoggerLike } from "../reporter/terminal.js";

export interface ListModelsDeps {
  cwd?: () => string;
  logger?: LoggerLike;
}

export function registerListModelsCommand(
  command: Command,
  deps: ListModelsDeps,
): void {
  const cwd = deps.cwd ?? (() => process.cwd());
  const logger = deps.logger ?? console;

  command
    .command("list-models")
    .description("List models discovered by setup")
    .action(async () => {
      const statePath = path.join(cwd(), ".lmstudio-bench.json");
      if (!(await fileExists(statePath))) {
        throw new Error("Run `lmstudio-bench setup` first");
      }
      const state = await readJson(statePath, LocalStateSchema);
      if (state.models.length === 0) {
        logger.log("No models discovered.");
        return;
      }
      for (const model of state.models) {
        logger.log(`${model.name}\t${model.path}`);
      }
    });
}
