import path from "path";
import type { Command } from "commander";
import { writeJson, writeYamlIfMissing } from "../config/loader.js";
import {
  LocalStateSchema,
  SweepConfigSchema,
} from "../config/schema.js";
import type { LocalState } from "../config/schema.js";
import { probeLlamaBenchCapabilities } from "../runner/llama-bench-capabilities.js";
import type { BenchCapabilities } from "../runner/llama-bench-capabilities.js";
import type { DiscoveryService } from "../bootstrap/discovery.js";
import type { LoggerLike } from "../reporter/terminal.js";

export interface SetupDeps {
  cwd?: () => string;
  logger?: LoggerLike;
  discoveryService: DiscoveryService;
  benchCapabilities?: BenchCapabilities;
}

export function createDefaultSweepConfig(
  state: LocalState,
  capabilities: BenchCapabilities,
) {
  const models =
    state.models.length > 0
      ? state.models
      : [{ name: "model", path: "<path-to-model.gguf>", size_gb: 0 }];
  const s = capabilities.supportedKeys;
  return SweepConfigSchema.parse({
    models: models.map((model) => ({
      path: model.path,
      name: model.name,
    })),
    baseline: {
      n_ctx: 4096,
      n_batch: 512,
      n_ubatch: 128,
      n_gpu_layers: 99,
      threads: 8,
      threads_batch: 8,
      kv_type_key: "f16",
      kv_type_value: "f16",
      flash_attention: true,
      mmap: true,
      mlock: false,
      no_kv_offload: false,
      n_cpu_moe: 0,
    },
    scan: {
      n_ctx: s.has("n_ctx") ? [2048, 4096, 8192] : [],
      n_batch: s.has("n_batch") ? [128, 256, 512, 1024] : [],
      n_ubatch: s.has("n_ubatch") ? [64, 128, 256, 512] : [],
      n_gpu_layers: s.has("n_gpu_layers")
        ? [0, 8, 16, 24, 30, 99]
        : [],
      threads: s.has("threads") ? [0, 4, 6, 8, 10, 12] : [],
      threads_batch: s.has("threads_batch")
        ? [0, 4, 6, 8, 10, 12]
        : [],
      kv_type_key: s.has("kv_type_key")
        ? [
            "f16",
            "f32",
            "s16",
            "q8_0",
            "q4_0",
            "q4_1",
            "iq4_nl",
            "q5_0",
            "q5_1",
          ]
        : [],
      kv_type_value: s.has("kv_type_value")
        ? [
            "f16",
            "f32",
            "s16",
            "q8_0",
            "q4_0",
            "q4_1",
            "iq4_nl",
            "q5_0",
            "q5_1",
          ]
        : [],
      flash_attention: s.has("flash_attention") ? [true, false] : [],
      mmap: s.has("mmap") ? [true, false] : [],
      mlock: s.has("mlock") ? [false, true] : [],
      no_kv_offload: s.has("no_kv_offload") ? [true, false] : [],
      n_cpu_moe: s.has("n_cpu_moe") ? [0, 1, 2, 4] : [],
    },
    workload: {
      prompt_tokens: [512],
      generation_tokens: [128],
      repetitions: 3,
    },
    model_overrides: [],
  });
}

export function registerSetupCommand(
  command: Command,
  deps: SetupDeps,
): void {
  const cwd = deps.cwd ?? (() => process.cwd());
  const logger = deps.logger ?? console;

  command
    .command("setup")
    .description("Discover llama.cpp tools and local models")
    .action(async () => {
      const [tools, hardware, models] = await Promise.all([
        deps.discoveryService.discoverTools(),
        deps.discoveryService.discoverHardware(),
        deps.discoveryService.discoverModels(),
      ]);
      const state = LocalStateSchema.parse({
        llama_bench_path: tools.llamaBenchPath,
        llama_cli_path: tools.llamaCliPath,
        hardware: {
          chip: hardware.chip,
          cores: hardware.cores,
          p_cores: hardware.pCores,
          e_cores: hardware.eCores,
          memory_gb: hardware.memoryGb,
          metal: hardware.metal,
        },
        model_dirs: models.modelDirs,
        models: models.models.map((model) => ({
          name: model.name,
          path: model.path,
          size_gb: model.sizeGb,
        })),
      });
      const statePath = path.join(cwd(), ".lmstudio-bench.json");
      const sweepConfigPath = path.join(cwd(), "sweep-config.yaml");
      await writeJson(statePath, state);
      const capabilities =
        deps.benchCapabilities ??
        (await probeLlamaBenchCapabilities(state.llama_bench_path));
      const wroteConfig = await writeYamlIfMissing(
        sweepConfigPath,
        createDefaultSweepConfig(state, capabilities),
      );
      logger.log(
        `Setup complete. Found ${state.models.length} model(s).`,
      );
      if (wroteConfig) {
        logger.log("Created sweep-config.yaml starter config.");
      }
    });
}
