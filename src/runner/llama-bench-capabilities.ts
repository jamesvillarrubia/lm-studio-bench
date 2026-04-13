import { runSubprocess } from "./subprocess.js";

export const BENCHMARK_TUNE_KEYS = [
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

export type BenchmarkTuneKey = (typeof BENCHMARK_TUNE_KEYS)[number];

export interface BenchCapabilities {
  supportedKeys: Set<string>;
  warnings: string[];
  usedFallback: boolean;
}

export const DEFAULT_FALLBACK_KEYS: ReadonlySet<string> = new Set<string>([
  "n_batch",
  "n_ubatch",
  "n_gpu_layers",
  "threads",
  "kv_type_key",
  "kv_type_value",
  "flash_attention",
  "mmap",
  "no_kv_offload",
]);

export function benchCapabilitiesFromKeys(
  keys: Iterable<string>,
  warnings: string[] = [],
  usedFallback = false,
): BenchCapabilities {
  return {
    supportedKeys: new Set(keys),
    warnings,
    usedFallback,
  };
}

export function benchCapabilitiesWithWarning(
  base: BenchCapabilities,
  message: string,
): BenchCapabilities {
  return benchCapabilitiesFromKeys(
    base.supportedKeys,
    [...base.warnings, message],
    base.usedFallback,
  );
}

export function extractLlamaBenchHelpSection(raw: string): string {
  const lines = raw.split(/\r?\n/);
  const usageIndex = lines.findIndex((line) => /^\s*usage:\s*/i.test(line));
  if (usageIndex >= 0) {
    return lines.slice(usageIndex).join("\n");
  }
  const testParamsIndex = lines.findIndex((line) =>
    /^\s*test parameters:\s*$/i.test(line),
  );
  if (testParamsIndex >= 0) {
    return lines.slice(testParamsIndex).join("\n");
  }
  return raw;
}

export function parseCapabilitiesFromHelpText(
  raw: string,
): BenchCapabilities {
  const help = extractLlamaBenchHelpSection(raw);
  const supported = new Set<string>();

  if (
    /(?:^|\n)\s*-b,\s*--batch-size\b/m.test(help) ||
    /\n\s*-b,\s*--batch-size\b/.test(help)
  ) {
    supported.add("n_batch");
  }
  if (
    /(?:^|\n)\s*-ub,\s*--ubatch-size\b/m.test(help) ||
    /\n\s*-ub,\s*--ubatch-size\b/.test(help)
  ) {
    supported.add("n_ubatch");
  }
  if (/(?:^|\n)\s*-ctk,\s*--cache-type-k\b/m.test(help)) {
    supported.add("kv_type_key");
  }
  if (/(?:^|\n)\s*-ctv,\s*--cache-type-v\b/m.test(help)) {
    supported.add("kv_type_value");
  }
  if (/(?:^|\n)\s*-t,\s*--threads\b/m.test(help)) {
    supported.add("threads");
  }
  if (/(?:^|\n)\s*-ngl,\s*--n-gpu-layers\b/m.test(help)) {
    supported.add("n_gpu_layers");
  }
  if (/(?:^|\n)\s*-fa,\s*--flash-attn\b/m.test(help)) {
    supported.add("flash_attention");
  }
  if (/(?:^|\n)\s*-mmp,\s*--mmap\b/m.test(help)) {
    supported.add("mmap");
  }
  if (
    /(?:^|\n)\s*-c,\s*--ctx-size\b/m.test(help) ||
    /(?:^|\n)\s*--ctx-size\b/m.test(help) ||
    /(?:^|\n)\s*-c,\s*--context-size\b/m.test(help)
  ) {
    supported.add("n_ctx");
  }
  if (
    /(?:^|\n)\s*-tb,\s*--threads-batch\b/m.test(help) ||
    /(?:^|\n)\s*--threads-batch\b/m.test(help)
  ) {
    supported.add("threads_batch");
  }
  if (
    /(?:^|\n)\s*-ml,\s*--mlock\b/m.test(help) ||
    /(?:^|\n)\s*--mlock\b/m.test(help)
  ) {
    supported.add("mlock");
  }
  if (/(?:^|\n)\s*-nkvo,\s*--no-kv-offload\b/m.test(help)) {
    supported.add("no_kv_offload");
  }
  if (/(?:^|\n)\s*-ncmoe,\s*--n-cpu-moe\b/m.test(help)) {
    supported.add("n_cpu_moe");
  }

  if (supported.size === 0) {
    return benchCapabilitiesFromKeys(
      DEFAULT_FALLBACK_KEYS,
      [
        "Could not infer any benchmark flags from llama-bench help; using conservative defaults.",
      ],
      true,
    );
  }
  return benchCapabilitiesFromKeys(supported, [], false);
}

export async function probeLlamaBenchCapabilities(
  llamaBenchPath: string,
): Promise<BenchCapabilities> {
  try {
    const result = await runSubprocess(llamaBenchPath, ["--help"]);
    const combined = `${result.stdout}\n${result.stderr}`;
    const helpBody = extractLlamaBenchHelpSection(combined);
    if (!helpBody.trim()) {
      return benchCapabilitiesFromKeys(
        DEFAULT_FALLBACK_KEYS,
        [
          `llama-bench --help produced no usable text (exit ${result.exitCode}); using defaults.`,
        ],
        true,
      );
    }
    const parsed = parseCapabilitiesFromHelpText(combined);
    if (result.exitCode !== 0 && parsed.warnings.length === 0) {
      return benchCapabilitiesWithWarning(
        parsed,
        `llama-bench --help exited with code ${result.exitCode}; capabilities may be incomplete.`,
      );
    }
    return parsed;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return benchCapabilitiesFromKeys(
      DEFAULT_FALLBACK_KEYS,
      [`Failed to run llama-bench --help (${message}); using defaults.`],
      true,
    );
  }
}
