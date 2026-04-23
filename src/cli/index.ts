import { Command } from "commander";
import { createDiscoveryService } from "../bootstrap/discovery.js";
import { createBenchRunner } from "../runner/llama-bench.js";
import type { BenchRunner } from "../runner/llama-bench.js";
import type { BenchCapabilities } from "../runner/llama-bench-capabilities.js";
import type { DiscoveryService } from "../bootstrap/discovery.js";
import type { LoggerLike } from "../reporter/terminal.js";
import { registerSetupCommand } from "./setup.js";
import { registerListModelsCommand } from "./list-models.js";
import { registerScanCommand } from "./scan.js";
import { registerCompareCommand } from "./compare.js";
import { registerReportCommand } from "./report.js";
import { registerVisualizeCommand } from "./visualize.js";
import { registerAnalyzeCommand } from "./analyze.js";

export interface CliDeps {
  logger?: LoggerLike & { error(message: string): void };
  benchRunner?: BenchRunner;
  benchCapabilities?: BenchCapabilities;
  discoveryService?: DiscoveryService;
}

const CLI_NAME = "lmstudio-bench";

export function createCli(deps: CliDeps = {}): Command {
  const command = new Command()
    .name(CLI_NAME)
    .description("Benchmark llama.cpp configurations");
  const logger = deps.logger ?? console;
  const benchRunner = deps.benchRunner ?? createBenchRunner();

  registerSetupCommand(command, {
    discoveryService:
      deps.discoveryService ?? createDiscoveryService(),
    ...(deps.benchCapabilities !== undefined
      ? { benchCapabilities: deps.benchCapabilities }
      : {}),
    logger,
  });
  registerListModelsCommand(command, {
    logger,
  });
  registerScanCommand(command, {
    benchRunner,
    ...(deps.benchCapabilities !== undefined
      ? { benchCapabilities: deps.benchCapabilities }
      : {}),
    logger,
  });
  registerCompareCommand(command, {
    benchRunner,
    ...(deps.benchCapabilities !== undefined
      ? { benchCapabilities: deps.benchCapabilities }
      : {}),
    logger,
  });
  registerReportCommand(command, {
    logger,
  });
  registerVisualizeCommand(command, {
    logger,
  });
  registerAnalyzeCommand(command, {
    logger,
  });

  return command;
}
