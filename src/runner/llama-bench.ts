import { runSubprocess } from "./subprocess.js";
import { inferLlamaBenchBinaryFingerprint } from "./bench-binary-fingerprint.js";
import { parseLlamaBenchOutput } from "./parser.js";
import type { BenchMetrics } from "./parser.js";
import {
  benchCapabilitiesFromKeys,
  DEFAULT_FALLBACK_KEYS,
} from "./llama-bench-capabilities.js";
import type { BenchCapabilities } from "./llama-bench-capabilities.js";

export interface BenchInput {
  modelPath: string;
  llamaBenchPath: string;
  config: Record<string, string | number | boolean>;
  capabilities?: BenchCapabilities;
  workload: {
    promptTokens: number;
    generationTokens: number;
  };
}

export interface BenchRunResult extends BenchMetrics {
  wallTimeSec: number;
}

export interface BenchRunner {
  getRunnerIdentity?(input: {
    llamaBenchPath: string;
  }): Promise<string>;
  run(input: BenchInput): Promise<BenchRunResult>;
}

export function buildLlamaBenchArgs(input: BenchInput): string[] {
  const caps =
    input.capabilities ??
    benchCapabilitiesFromKeys(DEFAULT_FALLBACK_KEYS, [], false);
  const args: string[] = [
    "-m",
    input.modelPath,
    "-p",
    String(input.workload.promptTokens),
    "-n",
    String(input.workload.generationTokens),
  ];

  if (
    caps.supportedKeys.has("n_ctx") &&
    typeof input.config.n_ctx === "number"
  ) {
    args.push("-c", String(input.config.n_ctx));
  }
  if (caps.supportedKeys.has("n_batch")) {
    args.push("-b", String(input.config.n_batch));
  }
  if (caps.supportedKeys.has("n_ubatch")) {
    args.push("-ub", String(input.config.n_ubatch));
  }
  if (caps.supportedKeys.has("n_gpu_layers")) {
    args.push("-ngl", String(input.config.n_gpu_layers));
  }
  if (caps.supportedKeys.has("threads")) {
    args.push("-t", String(input.config.threads));
  }
  if (
    caps.supportedKeys.has("threads_batch") &&
    typeof input.config.threads_batch === "number"
  ) {
    args.push("-tb", String(input.config.threads_batch));
  }
  if (caps.supportedKeys.has("kv_type_key")) {
    args.push("-ctk", String(input.config.kv_type_key));
  }
  if (caps.supportedKeys.has("kv_type_value")) {
    args.push("-ctv", String(input.config.kv_type_value));
  }
  if (caps.supportedKeys.has("flash_attention")) {
    args.push("-fa", input.config.flash_attention ? "1" : "0");
  }
  if (caps.supportedKeys.has("mmap")) {
    args.push("-mmp", input.config.mmap ? "1" : "0");
  }
  if (caps.supportedKeys.has("mlock")) {
    args.push("--mlock", input.config.mlock ? "1" : "0");
  }
  if (
    caps.supportedKeys.has("no_kv_offload") &&
    typeof input.config.no_kv_offload === "boolean"
  ) {
    args.push("-nkvo", input.config.no_kv_offload ? "1" : "0");
  }
  if (
    caps.supportedKeys.has("n_cpu_moe") &&
    typeof input.config.n_cpu_moe === "number"
  ) {
    args.push("-ncmoe", String(input.config.n_cpu_moe));
  }

  args.push("-r", "1", "-o", "json");
  return args;
}

class LlamaBenchRunner implements BenchRunner {
  private identityCache = new Map<string, string>();

  async getRunnerIdentity(input: {
    llamaBenchPath: string;
  }): Promise<string> {
    const cached = this.identityCache.get(input.llamaBenchPath);
    if (cached) {
      return cached;
    }
    const identity = await inferLlamaBenchBinaryFingerprint(
      input.llamaBenchPath,
    );
    this.identityCache.set(input.llamaBenchPath, identity);
    return identity;
  }

  async run(input: BenchInput): Promise<BenchRunResult> {
    const args = buildLlamaBenchArgs(input);
    const result = await runSubprocess(input.llamaBenchPath, args);
    if (result.exitCode !== 0) {
      throw new Error(
        result.stderr ||
          `llama-bench failed with exit code ${result.exitCode}`,
      );
    }
    const parsed = parseLlamaBenchOutput(result.stdout);
    return {
      ...parsed,
      wallTimeSec: Number((result.elapsedMs / 1e3).toFixed(3)),
    };
  }
}

export function createBenchRunner(): BenchRunner {
  return new LlamaBenchRunner();
}

export function createBenchInput(
  state: { llama_bench_path: string },
  modelPath: string,
  config: Record<string, string | number | boolean>,
  promptTokens: number,
  generationTokens: number,
  capabilities?: BenchCapabilities,
): BenchInput {
  return {
    modelPath,
    llamaBenchPath: state.llama_bench_path,
    config,
    ...(capabilities !== undefined ? { capabilities } : {}),
    workload: {
      promptTokens,
      generationTokens,
    },
  };
}
