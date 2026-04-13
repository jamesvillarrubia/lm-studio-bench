#!/usr/bin/env node

// src/cli/index.ts
import { Command } from "commander";

// src/bootstrap/discovery.ts
import { access, readdir, stat } from "fs/promises";
import { constants } from "fs";
import os from "os";
import path from "path";
var DEFAULT_MODEL_DIRS = ["~/.lmstudio/models", "~/Library/Application Support/LM Studio/models"];
function expandHome(dir) {
  if (!dir.startsWith("~/")) {
    return dir;
  }
  return path.join(os.homedir(), dir.slice(2));
}
async function isExecutable(filePath) {
  try {
    await access(filePath, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}
async function findOnPath(binaryName) {
  const envPath = process.env.PATH ?? "";
  const segments = envPath.split(path.delimiter).filter(Boolean);
  for (const segment of segments) {
    const candidate = path.join(segment, binaryName);
    if (await isExecutable(candidate)) {
      return candidate;
    }
  }
  return null;
}
function isRunnableModelFile(fileName) {
  if (!fileName.endsWith(".gguf")) {
    return false;
  }
  const lower = fileName.toLowerCase();
  if (lower.startsWith("mmproj-") || lower.includes("-mmproj-") || lower.includes("mmproj")) {
    return false;
  }
  return true;
}
async function walkForGgufFiles(root) {
  const expandedRoot = expandHome(root);
  try {
    const entries = await readdir(expandedRoot, { withFileTypes: true, recursive: true });
    const models = [];
    for (const entry of entries) {
      if (!entry.isFile() || !isRunnableModelFile(entry.name)) {
        continue;
      }
      const fullPath = path.join(entry.parentPath, entry.name);
      const fileStats = await stat(fullPath);
      models.push({
        name: entry.name.replace(/\.gguf$/i, ""),
        path: fullPath,
        sizeGb: Number((fileStats.size / 1024 / 1024 / 1024).toFixed(2))
      });
    }
    return models;
  } catch {
    return [];
  }
}
function createDiscoveryService() {
  return {
    async discoverTools() {
      const llamaBenchPath = await findOnPath("llama-bench");
      const llamaCliPath = await findOnPath("llama-cli");
      if (!llamaBenchPath || !llamaCliPath) {
        throw new Error("Could not find llama-bench and llama-cli on PATH");
      }
      return { llamaBenchPath, llamaCliPath };
    },
    async discoverHardware() {
      const cores = os.cpus().length;
      return {
        chip: process.platform === "darwin" ? "Apple Silicon" : process.arch,
        cores,
        pCores: Math.max(1, Math.ceil(cores * 0.66)),
        eCores: Math.max(0, cores - Math.ceil(cores * 0.66)),
        memoryGb: Math.round(os.totalmem() / 1024 / 1024 / 1024),
        metal: process.platform === "darwin"
      };
    },
    async discoverModels() {
      const modelBatches = await Promise.all(DEFAULT_MODEL_DIRS.map((dir) => walkForGgufFiles(dir)));
      return {
        modelDirs: DEFAULT_MODEL_DIRS,
        models: modelBatches.flat()
      };
    }
  };
}

// src/runner/bench-binary-fingerprint.ts
import { createHash } from "crypto";
import { open, stat as stat2 } from "fs/promises";
var HEAD_BYTES = 1024 * 1024;
async function inferLlamaBenchBinaryFingerprint(llamaBenchPath) {
  try {
    const fileStats = await stat2(llamaBenchPath);
    const file = await open(llamaBenchPath, "r");
    try {
      const buffer = Buffer.allocUnsafe(Math.min(HEAD_BYTES, Number(fileStats.size)));
      if (buffer.length === 0) {
        return `llama_bench_binary|path:${llamaBenchPath}|size:0|mtime_ms:${Number(fileStats.mtimeMs)}|head_sha256:empty`;
      }
      const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
      const head = buffer.subarray(0, bytesRead);
      const headSha256 = createHash("sha256").update(head).digest("hex");
      return `llama_bench_binary|path:${llamaBenchPath}|size:${fileStats.size}|mtime_ms:${Number(
        fileStats.mtimeMs
      )}|head_sha256:${headSha256}`;
    } finally {
      await file.close();
    }
  } catch {
    return `llama_bench_binary|missing_path:${llamaBenchPath}`;
  }
}

// src/runner/subprocess.ts
import { spawn } from "child_process";
async function runSubprocess(command, args) {
  const start = Date.now();
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (exitCode) => {
      resolve({
        stdout,
        stderr,
        exitCode: exitCode ?? 1,
        elapsedMs: Date.now() - start
      });
    });
  });
}

// src/runner/llama-bench-capabilities.ts
var BENCHMARK_TUNE_KEYS = [
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
  "mlock"
];
var DEFAULT_FALLBACK_KEYS = /* @__PURE__ */ new Set([
  "n_batch",
  "n_ubatch",
  "n_gpu_layers",
  "threads",
  "kv_type_key",
  "kv_type_value",
  "flash_attention",
  "mmap"
]);
function benchCapabilitiesFromKeys(keys, warnings = [], usedFallback = false) {
  return {
    supportedKeys: new Set(keys),
    warnings,
    usedFallback
  };
}
function benchCapabilitiesWithWarning(base, message) {
  return benchCapabilitiesFromKeys(base.supportedKeys, [...base.warnings, message], base.usedFallback);
}
function extractLlamaBenchHelpSection(raw) {
  const lines = raw.split(/\r?\n/);
  const usageIndex = lines.findIndex((line) => /^\s*usage:\s*/i.test(line));
  if (usageIndex >= 0) {
    return lines.slice(usageIndex).join("\n");
  }
  const testParamsIndex = lines.findIndex((line) => /^\s*test parameters:\s*$/i.test(line));
  if (testParamsIndex >= 0) {
    return lines.slice(testParamsIndex).join("\n");
  }
  return raw;
}
function parseCapabilitiesFromHelpText(raw) {
  const help = extractLlamaBenchHelpSection(raw);
  const supported = /* @__PURE__ */ new Set();
  if (/(?:^|\n)\s*-b,\s*--batch-size\b/m.test(help) || /\n\s*-b,\s*--batch-size\b/.test(help)) {
    supported.add("n_batch");
  }
  if (/(?:^|\n)\s*-ub,\s*--ubatch-size\b/m.test(help) || /\n\s*-ub,\s*--ubatch-size\b/.test(help)) {
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
  if (/(?:^|\n)\s*-c,\s*--ctx-size\b/m.test(help) || /(?:^|\n)\s*--ctx-size\b/m.test(help) || /(?:^|\n)\s*-c,\s*--context-size\b/m.test(help)) {
    supported.add("n_ctx");
  }
  if (/(?:^|\n)\s*-tb,\s*--threads-batch\b/m.test(help) || /(?:^|\n)\s*--threads-batch\b/m.test(help)) {
    supported.add("threads_batch");
  }
  if (/(?:^|\n)\s*-ml,\s*--mlock\b/m.test(help) || /(?:^|\n)\s*--mlock\b/m.test(help)) {
    supported.add("mlock");
  }
  if (supported.size === 0) {
    return benchCapabilitiesFromKeys(
      DEFAULT_FALLBACK_KEYS,
      ["Could not infer any benchmark flags from llama-bench help; using conservative defaults."],
      true
    );
  }
  return benchCapabilitiesFromKeys(supported, [], false);
}
async function probeLlamaBenchCapabilities(llamaBenchPath) {
  try {
    const result = await runSubprocess(llamaBenchPath, ["--help"]);
    const combined = `${result.stdout}
${result.stderr}`;
    const helpBody = extractLlamaBenchHelpSection(combined);
    if (!helpBody.trim()) {
      return benchCapabilitiesFromKeys(
        DEFAULT_FALLBACK_KEYS,
        [`llama-bench --help produced no usable text (exit ${result.exitCode}); using defaults.`],
        true
      );
    }
    const parsed = parseCapabilitiesFromHelpText(combined);
    if (result.exitCode !== 0 && parsed.warnings.length === 0) {
      return benchCapabilitiesWithWarning(
        parsed,
        `llama-bench --help exited with code ${result.exitCode}; capabilities may be incomplete.`
      );
    }
    return parsed;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return benchCapabilitiesFromKeys(
      DEFAULT_FALLBACK_KEYS,
      [`Failed to run llama-bench --help (${message}); using defaults.`],
      true
    );
  }
}

// src/runner/parser.ts
function parseLegacyJsonObject(parsed) {
  const pp = Number(parsed.pp_tokens_per_sec);
  const tg = Number(parsed.tg_tokens_per_sec);
  const ttftValue = Number(parsed.ttft_ms);
  const ttft = Number.isFinite(ttftValue) ? ttftValue : null;
  if ([pp, tg].every(Number.isFinite)) {
    return {
      ppTokensPerSec: pp,
      tgTokensPerSec: tg,
      ttftMs: ttft
    };
  }
  return null;
}
function parseModernJsonArray(parsed) {
  const promptEntry = parsed.find(
    (entry) => typeof entry === "object" && entry !== null && Number(entry.n_prompt) > 0 && Number(entry.n_gen) === 0
  );
  const generationEntry = parsed.find(
    (entry) => typeof entry === "object" && entry !== null && Number(entry.n_gen) > 0 && Number(entry.n_prompt) === 0
  );
  const pp = Number(promptEntry?.avg_ts);
  const tg = Number(generationEntry?.avg_ts);
  if ([pp, tg].every(Number.isFinite)) {
    return {
      ppTokensPerSec: pp,
      tgTokensPerSec: tg,
      ttftMs: null
    };
  }
  return null;
}
function parseJsonOutput(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parseModernJsonArray(parsed);
    }
    if (parsed && typeof parsed === "object") {
      return parseLegacyJsonObject(parsed);
    }
    return null;
  } catch {
    return null;
  }
}
function parseMetricByRegex(raw, patterns) {
  for (const pattern of patterns) {
    const match = raw.match(pattern);
    const value = Number(match?.[1] ?? Number.NaN);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}
function parseLlamaBenchOutput(raw) {
  const fromJson = parseJsonOutput(raw);
  if (fromJson) {
    return fromJson;
  }
  const pp = parseMetricByRegex(raw, [
    /pp(?:_tokens_per_sec)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
    /prompt(?:\s+processing)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i
  ]);
  const tg = parseMetricByRegex(raw, [
    /tg(?:_tokens_per_sec)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
    /generation(?:\s+speed)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i
  ]);
  const ttft = parseMetricByRegex(raw, [
    /ttft(?:_ms)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
    /time\s+to\s+first\s+token\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i
  ]);
  if (pp === null || tg === null) {
    throw new Error("Unable to parse llama-bench output");
  }
  return {
    ppTokensPerSec: pp,
    tgTokensPerSec: tg,
    ttftMs: ttft ?? null
  };
}

// src/runner/llama-bench.ts
function buildLlamaBenchArgs(input) {
  const caps = input.capabilities ?? benchCapabilitiesFromKeys(DEFAULT_FALLBACK_KEYS, [], false);
  const args = [
    "-m",
    input.modelPath,
    "-p",
    String(input.workload.promptTokens),
    "-n",
    String(input.workload.generationTokens)
  ];
  if (caps.supportedKeys.has("n_ctx") && typeof input.config.n_ctx === "number") {
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
  if (caps.supportedKeys.has("threads_batch") && typeof input.config.threads_batch === "number") {
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
  args.push("-r", "1", "-o", "json");
  return args;
}
var LlamaBenchRunner = class {
  identityCache = /* @__PURE__ */ new Map();
  async getRunnerIdentity(input) {
    const cached = this.identityCache.get(input.llamaBenchPath);
    if (cached) {
      return cached;
    }
    const identity = await inferLlamaBenchBinaryFingerprint(input.llamaBenchPath);
    this.identityCache.set(input.llamaBenchPath, identity);
    return identity;
  }
  async run(input) {
    const args = buildLlamaBenchArgs(input);
    const result = await runSubprocess(input.llamaBenchPath, args);
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `llama-bench failed with exit code ${result.exitCode}`);
    }
    const parsed = parseLlamaBenchOutput(result.stdout);
    return {
      ...parsed,
      wallTimeSec: Number((result.elapsedMs / 1e3).toFixed(3))
    };
  }
};
function createBenchRunner() {
  return new LlamaBenchRunner();
}
function createBenchInput(state, modelPath, config, promptTokens, generationTokens, capabilities) {
  return {
    modelPath,
    llamaBenchPath: state.llama_bench_path,
    config,
    ...capabilities !== void 0 ? { capabilities } : {},
    workload: {
      promptTokens,
      generationTokens
    }
  };
}

// src/cli/compare.ts
import { writeFile as writeFile3 } from "fs/promises";
import path3 from "path";

// src/cache/run-key.ts
import { createHash as createHash2 } from "crypto";
function stableStringify(value) {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  const entries = Object.entries(value).filter(([, v]) => typeof v !== "undefined").sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0);
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(",")}}`;
}
function computeRunKey(input) {
  const payload = stableStringify({
    v: 1,
    benchCommand: input.benchCommand,
    modelPath: input.modelPath,
    appliedConfig: input.appliedConfig,
    workload: input.workload,
    repetition: input.repetition,
    hardware: input.hardware,
    runnerIdentity: input.runnerIdentity
  });
  return createHash2("sha256").update(payload, "utf8").digest("hex");
}

// src/config/schema.ts
import { z } from "zod";
var LocalModelSchema = z.object({
  name: z.string().min(1),
  path: z.string().min(1),
  size_gb: z.number().nonnegative()
});
var LocalStateSchema = z.object({
  llama_bench_path: z.string().min(1),
  llama_cli_path: z.string().min(1),
  hardware: z.object({
    chip: z.string().min(1),
    cores: z.number().int().positive(),
    p_cores: z.number().int().nonnegative(),
    e_cores: z.number().int().nonnegative(),
    memory_gb: z.number().int().positive(),
    metal: z.boolean()
  }),
  model_dirs: z.array(z.string().min(1)).default([]),
  models: z.array(LocalModelSchema).default([])
});
var SweepModelSchema = z.object({
  path: z.string().min(1),
  name: z.string().min(1)
});
var KvCacheTypeSchema = z.string().min(1);
var BenchmarkBaselineSchema = z.object({
  n_ctx: z.number().int().positive(),
  n_batch: z.number().int().positive(),
  n_ubatch: z.number().int().positive(),
  n_gpu_layers: z.number().int().nonnegative(),
  threads: z.number().int().nonnegative(),
  threads_batch: z.number().int().nonnegative(),
  kv_type_key: KvCacheTypeSchema,
  kv_type_value: KvCacheTypeSchema,
  flash_attention: z.boolean(),
  mmap: z.boolean(),
  mlock: z.boolean()
});
var BenchmarkScanSchema = z.object({
  n_ctx: z.array(z.number().int().positive()).default([]),
  n_batch: z.array(z.number().int().positive()).default([]),
  n_ubatch: z.array(z.number().int().positive()).default([]),
  n_gpu_layers: z.array(z.number().int().nonnegative()).default([]),
  threads: z.array(z.number().int().nonnegative()).default([]),
  threads_batch: z.array(z.number().int().nonnegative()).default([]),
  kv_type_key: z.array(KvCacheTypeSchema).default([]),
  kv_type_value: z.array(KvCacheTypeSchema).default([]),
  flash_attention: z.array(z.boolean()).default([]),
  mmap: z.array(z.boolean()).default([]),
  mlock: z.array(z.boolean()).default([])
});
var WorkloadSchema = z.object({
  prompt_tokens: z.array(z.number().int().positive()).default([512]),
  generation_tokens: z.array(z.number().int().positive()).default([128]),
  repetitions: z.number().int().positive().default(3)
});
var SweepConfigSchema = z.object({
  models: z.array(SweepModelSchema).min(1),
  baseline: BenchmarkBaselineSchema,
  scan: BenchmarkScanSchema,
  workload: WorkloadSchema,
  model_overrides: z.array(
    z.object({
      match: z.object({
        name: z.string().min(1).optional(),
        path: z.string().min(1).optional()
      }).refine((value) => typeof value.name === "string" || typeof value.path === "string", {
        message: "model_overrides.match requires at least one of name or path"
      }),
      baseline: BenchmarkBaselineSchema.partial().optional(),
      scan: BenchmarkScanSchema.partial().optional(),
      workload: WorkloadSchema.partial().optional()
    })
  ).default([])
});
var runRecordScalar = z.union([z.string(), z.number(), z.boolean()]);
var RunRecordSchema = z.object({
  id: z.string().min(1),
  timestamp: z.string().min(1),
  model: z.string().min(1),
  /**
   * Stable identity for exact-match caching across `scan` / `compare`.
   *
   * Older JSONL lines may omit this field; those records are never treated as cache hits.
   */
  run_key: z.string().min(1).optional(),
  bench_command: z.enum(["scan", "compare"]).optional(),
  runner_identity: z.string().min(1).optional(),
  hardware: LocalStateSchema.shape.hardware.optional(),
  cache: z.object({
    /**
     * When true, this record was synthesized from a prior successful JSONL record and
     * did not execute a fresh benchmark in this session.
     */
    hit: z.boolean(),
    /**
     * The `id` of the prior record this cache hit reused (best-effort; omitted for legacy hits).
     */
    reused_run_id: z.string().min(1).optional()
  }).optional(),
  /**
   * YAML / sweep row as interpreted for this run (may include fields not passed to `llama-bench`).
   */
  requested_config: z.record(z.string(), runRecordScalar).optional(),
  /**
   * Subset of `requested_config` that was actually passed to `llama-bench` for this run.
   */
  applied_config: z.record(z.string(), runRecordScalar).optional(),
  capability_meta: z.object({
    supported_keys: z.array(z.string()),
    used_fallback: z.boolean(),
    warnings: z.array(z.string()).optional()
  }).optional(),
  /** @deprecated Prefer `applied_config`; retained for older JSONL rows. */
  config: z.record(z.string(), runRecordScalar).optional(),
  workload: z.object({
    pp: z.number().int().positive(),
    tg: z.number().int().positive()
  }),
  metrics: z.object({
    pp_tokens_per_sec: z.number().nullable(),
    tg_tokens_per_sec: z.number().nullable(),
    ttft_ms: z.number().nullable(),
    peak_memory_mb: z.number().nullable().optional(),
    load_time_ms: z.number().nullable().optional(),
    wall_time_sec: z.number().nullable().optional()
  }),
  run_index: z.number().int().positive(),
  repetitions: z.number().int().positive(),
  status: z.enum(["success", "failed"]),
  error: z.object({
    message: z.string().min(1),
    code: z.string().optional()
  }).nullable()
}).superRefine((value, ctx) => {
  if (!value.config && !value.applied_config) {
    ctx.addIssue({
      code: "custom",
      message: "RunRecord requires config (legacy) or applied_config",
      path: ["applied_config"]
    });
  }
});

// src/config/resolution.ts
function applyModelOverrides(config, model) {
  const matchedOverrides = config.model_overrides.filter((override) => {
    const nameMatch = override.match.name ? override.match.name === model.name : true;
    const pathMatch = override.match.path ? override.match.path === model.path : true;
    return nameMatch && pathMatch;
  });
  let baseline = { ...config.baseline };
  let scan = { ...config.scan };
  let workload = { ...config.workload };
  for (const override of matchedOverrides) {
    baseline = BenchmarkBaselineSchema.parse({ ...baseline, ...override.baseline ?? {} });
    scan = BenchmarkScanSchema.parse({ ...scan, ...override.scan ?? {} });
    workload = WorkloadSchema.parse({ ...workload, ...override.workload ?? {} });
  }
  return {
    baseline,
    scan,
    workload
  };
}
function isRecordKey(key) {
  return key in BenchmarkBaselineSchema.shape;
}
function filterToAppliedBenchmarkConfig(source, capabilities) {
  const out = {};
  for (const key of BENCHMARK_TUNE_KEYS) {
    if (!capabilities.supportedKeys.has(key)) {
      continue;
    }
    if (typeof source === "object" && source !== null && key in source) {
      out[key] = source[key];
    }
  }
  return out;
}
function filterScanForCapabilities(scan, capabilities) {
  return Object.fromEntries(
    Object.entries(scan).filter(([key]) => {
      if (!isRecordKey(key)) {
        return false;
      }
      return capabilities.supportedKeys.has(key);
    })
  );
}
function findUnsupportedScanKeys(scan, capabilities) {
  return Object.keys(scan).filter((key) => {
    if (!isRecordKey(key)) {
      return false;
    }
    const values = scan[key] ?? [];
    return Array.isArray(values) && values.length > 0 && !capabilities.supportedKeys.has(key);
  });
}
function findUnsupportedBaselineKeys(baseline, capabilities) {
  return Object.keys(baseline).filter(
    (key) => isRecordKey(key) && !capabilities.supportedKeys.has(key)
  );
}

// src/config/loader.ts
import { access as access2, readFile, writeFile } from "fs/promises";
import { constants as constants2 } from "fs";
import { parse, stringify } from "yaml";
async function fileExists(filePath) {
  try {
    await access2(filePath, constants2.F_OK);
    return true;
  } catch {
    return false;
  }
}
async function writeJson(filePath, value) {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}
`, "utf8");
}
async function readJson(filePath, schema) {
  const raw = await readFile(filePath, "utf8");
  const parsed = JSON.parse(raw);
  return schema.parse(parsed);
}
async function writeYaml(filePath, value) {
  await writeFile(filePath, stringify(value), "utf8");
}
async function readYaml(filePath, schema) {
  const raw = await readFile(filePath, "utf8");
  const parsed = parse(raw);
  return schema.parse(parsed);
}
async function writeYamlIfMissing(filePath, value) {
  if (await fileExists(filePath)) {
    return false;
  }
  await writeYaml(filePath, value);
  return true;
}

// src/reporter/csv.ts
function escapeCsvCell(value) {
  if (value === null) {
    return "";
  }
  const raw = String(value);
  if (raw.includes(",") || raw.includes('"') || raw.includes("\n")) {
    return `"${raw.replaceAll('"', '""')}"`;
  }
  return raw;
}
function buildModelSummaryCsv(summaries) {
  const header = "rank,median_tg_tokens_per_sec,median_pp_tokens_per_sec,median_ttft_ms,success_count,failed_count,config_key";
  const lines = summaries.map(
    (summary, index) => [
      index + 1,
      summary.medianTg,
      summary.medianPp,
      summary.medianTtft,
      summary.successCount,
      summary.failedCount,
      summary.configKey
    ].map((entry) => escapeCsvCell(entry)).join(",")
  );
  return [header, ...lines].join("\n").concat("\n");
}
function buildComparisonCsv(rows) {
  const header = "model,best_median_tg_tokens_per_sec,best_median_pp_tokens_per_sec,best_median_ttft_ms,best_config_key";
  const lines = rows.map(
    (row) => [row.model, row.bestMedianTg, row.bestMedianPp, row.bestMedianTtft, row.bestConfigKey].map((entry) => escapeCsvCell(entry)).join(",")
  );
  return [header, ...lines].join("\n").concat("\n");
}

// src/reporter/results.ts
import { mkdir, readFile as readFile2, writeFile as writeFile2 } from "fs/promises";
import path2 from "path";
function median(values) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    const left = sorted[middle - 1];
    const right = sorted[middle];
    if (typeof left !== "number" || typeof right !== "number") {
      return null;
    }
    return Number(((left + right) / 2).toFixed(3));
  }
  const single = sorted[middle];
  if (typeof single !== "number") {
    return null;
  }
  return Number(single.toFixed(3));
}
function recordAppliedConfig(record) {
  return record.applied_config ?? record.config;
}
function configKey(config) {
  const sorted = Object.keys(config).sort().map((key) => [key, config[key]]);
  return JSON.stringify(sorted);
}
async function appendRunRecord(runsFilePath, runRecord) {
  await mkdir(path2.dirname(runsFilePath), { recursive: true });
  const parsed = RunRecordSchema.parse(runRecord);
  await writeFile2(runsFilePath, `${JSON.stringify(parsed)}
`, { encoding: "utf8", flag: "a" });
}
function parseRunRecordsFromJsonl(raw) {
  const records = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    let json;
    try {
      json = JSON.parse(trimmed);
    } catch {
      continue;
    }
    const parsed = RunRecordSchema.safeParse(json);
    if (parsed.success) {
      records.push(parsed.data);
    }
  }
  return records;
}
function parseRunRecordsFromJsonlWithDiagnostics(raw) {
  const records = [];
  let invalidLineCount = 0;
  let hadAnyNonEmptyLine = false;
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    hadAnyNonEmptyLine = true;
    let json;
    try {
      json = JSON.parse(trimmed);
    } catch {
      invalidLineCount += 1;
      continue;
    }
    const parsed = RunRecordSchema.safeParse(json);
    if (parsed.success) {
      records.push(parsed.data);
    } else {
      invalidLineCount += 1;
    }
  }
  return { records, hadAnyNonEmptyLine, invalidLineCount };
}
async function loadRunRecords(runsFilePath) {
  try {
    const raw = await readFile2(runsFilePath, "utf8");
    return parseRunRecordsFromJsonl(raw);
  } catch (error) {
    const code = error.code;
    if (code === "ENOENT") {
      return [];
    }
    throw error;
  }
}
async function loadRunRecordsWithDiagnostics(runsFilePath) {
  try {
    const raw = await readFile2(runsFilePath, "utf8");
    return parseRunRecordsFromJsonlWithDiagnostics(raw);
  } catch (error) {
    const code = error.code;
    if (code === "ENOENT") {
      return { records: [], hadAnyNonEmptyLine: false, invalidLineCount: 0 };
    }
    throw error;
  }
}
function findLatestRunRecordByRunKey(records, runKey) {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const record = records[index];
    if (!record) {
      continue;
    }
    if (record.run_key === runKey) {
      return record;
    }
  }
  return null;
}
function summarizeRecords(records) {
  const byConfig = /* @__PURE__ */ new Map();
  for (const record of records) {
    const applied = recordAppliedConfig(record);
    const key = configKey(applied);
    const existing = byConfig.get(key) ?? [];
    existing.push(record);
    byConfig.set(key, existing);
  }
  const summaries = [];
  for (const [key, grouped] of byConfig.entries()) {
    const successes = grouped.filter((item) => item.status === "success");
    const failures = grouped.filter((item) => item.status === "failed");
    const ppValues = successes.map((item) => item.metrics.pp_tokens_per_sec).filter((value) => typeof value === "number");
    const tgValues = successes.map((item) => item.metrics.tg_tokens_per_sec).filter((value) => typeof value === "number");
    const ttftValues = successes.map((item) => item.metrics.ttft_ms).filter((value) => typeof value === "number");
    summaries.push({
      configKey: key,
      config: grouped[0] ? recordAppliedConfig(grouped[0]) : {},
      successCount: successes.length,
      failedCount: failures.length,
      medianPp: median(ppValues),
      medianTg: median(tgValues),
      medianTtft: median(ttftValues)
    });
  }
  return summaries.sort((left, right) => (right.medianTg ?? -1) - (left.medianTg ?? -1));
}

// src/reporter/terminal.ts
import chalk from "chalk";
import Table from "cli-table3";
function formatScalarForBaseline(value) {
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return String(value);
  }
  return JSON.stringify(value);
}
function formatOptimalSettingsLines(config) {
  const keys = Object.keys(config).sort();
  if (keys.length === 0) {
    return "  (no applied fields in summary)";
  }
  return keys.map((key) => {
    const value = config[key];
    if (value === void 0) {
      return `  ${key}: (missing)`;
    }
    return `  ${key}: ${formatScalarForBaseline(value)}`;
  }).join("\n");
}
function printScanSummary(logger, summaries, options) {
  if (summaries.length === 0) {
    logger.log("No scan results to summarize.");
    return;
  }
  const table = new Table({
    head: ["rank", "tg tokens/s", "pp tokens/s", "ttft ms", "success", "failed", "config"]
  });
  summaries.forEach((summary, index) => {
    table.push([
      index + 1,
      summary.medianTg ?? "-",
      summary.medianPp ?? "-",
      summary.medianTtft ?? "-",
      summary.successCount,
      summary.failedCount,
      summary.configKey
    ]);
  });
  logger.log(table.toString());
  const best = summaries[0];
  if (!best) {
    return;
  }
  logger.log(chalk.green(`Best throughput: rank #1 (tg=${best.medianTg ?? "n/a"} tokens/s)`));
  const modelLabel = options?.modelName ? ` for ${options.modelName}` : "";
  logger.log("");
  if (best.successCount === 0) {
    logger.log(
      chalk.yellow(
        `No successful runs${modelLabel}; the block below is still rank #1 by ordering but medians are missing \u2014 fix failures and re-scan.`
      )
    );
  } else {
    logger.log(chalk.bold(`Optimal benchmark settings${modelLabel} (rank #1 by median tg tokens/s):`));
  }
  logger.log(
    chalk.dim("Paste under baseline in sweep-config (these are applied llama-bench fields for this build):")
  );
  logger.log(formatOptimalSettingsLines(best.config));
}
function printModelComparison(logger, rows) {
  logger.log("Model comparison");
  if (rows.length === 0) {
    logger.log("No model rows available.");
    return;
  }
  const table = new Table({
    head: ["model", "best tg", "best pp", "best ttft", "config"]
  });
  rows.forEach((row) => {
    table.push([row.model, row.bestMedianTg ?? "-", row.bestMedianPp ?? "-", row.bestMedianTtft ?? "-", row.bestConfigKey]);
  });
  logger.log(table.toString());
}

// src/cli/compare.ts
function collectString(value, previous) {
  return [...previous, value];
}
function sanitizeFilePart(value) {
  return value.replace(/[^a-zA-Z0-9._-]/g, "-");
}
function tuningSnapshotToRecord(source) {
  const out = {};
  for (const [key, value] of Object.entries(source)) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      out[key] = value;
    }
  }
  return out;
}
function resolveModelPath(state, modelRef, configModels) {
  if (modelRef.includes("/") || modelRef.endsWith(".gguf")) {
    return { modelName: path3.basename(modelRef, path3.extname(modelRef)), modelPath: modelRef };
  }
  const fromConfig = configModels.find((item) => item.name === modelRef);
  if (fromConfig) {
    return { modelName: fromConfig.name, modelPath: fromConfig.path };
  }
  const fromState = state.models.find((item) => item.name === modelRef);
  if (fromState) {
    return { modelName: fromState.name, modelPath: fromState.path };
  }
  throw new Error(`Model '${modelRef}' could not be resolved`);
}
function buildCacheHitRecord(base, reused) {
  return RunRecordSchema.parse({
    ...base,
    metrics: reused.metrics,
    status: "success",
    error: null,
    cache: {
      hit: true,
      reused_run_id: reused.id
    }
  });
}
function registerCompareCommand(command, deps) {
  const cwd = deps.cwd ?? (() => process.cwd());
  const logger = deps.logger ?? console;
  const now = deps.now ?? (() => /* @__PURE__ */ new Date());
  command.command("compare").description("Benchmark models head-to-head on baseline configuration").requiredOption("-m, --model <model>", "Model name or path (repeat for multiple models)", collectString, []).option("-c, --config <path>", "Config path", "sweep-config.yaml").option("--rerun", "Ignore successful benchmark cache hits and execute all planned runs").option("--retry-failed", "Skip successful cache hits, but rerun failures (based on latest record per run key)").action(async (options) => {
    if (options.model.length < 2) {
      throw new Error("Provide at least two --model values for comparison");
    }
    const root = cwd();
    const statePath = path3.join(root, ".lmstudio-bench.json");
    if (!await fileExists(statePath)) {
      throw new Error("Run `lmstudio-bench setup` first");
    }
    const configPath = path3.join(root, options.config ?? "sweep-config.yaml");
    const state = await readJson(statePath, LocalStateSchema);
    const capabilities = deps.benchCapabilities ?? await probeLlamaBenchCapabilities(state.llama_bench_path);
    for (const warning of capabilities.warnings) {
      logger.log(`llama-bench capabilities: ${warning}`);
    }
    const config = await readYaml(configPath, SweepConfigSchema);
    const runnerIdentity = await deps.benchRunner.getRunnerIdentity?.({ llamaBenchPath: state.llama_bench_path }) ?? await inferLlamaBenchBinaryFingerprint(state.llama_bench_path);
    const rows = [];
    let hadFailures = false;
    for (const modelRef of options.model) {
      const model = resolveModelPath(state, modelRef, config.models);
      const resolvedConfig = applyModelOverrides(config, {
        name: model.modelName,
        path: model.modelPath
      });
      const unsupportedBaselineKeys = findUnsupportedBaselineKeys(resolvedConfig.baseline, capabilities);
      if (unsupportedBaselineKeys.length > 0) {
        logger.log(
          `Ignoring unsupported baseline fields for ${model.modelName}: ${unsupportedBaselineKeys.join(", ")}`
        );
      }
      const promptTokens = resolvedConfig.workload.prompt_tokens[0];
      const generationTokens = resolvedConfig.workload.generation_tokens[0];
      if (typeof promptTokens !== "number" || typeof generationTokens !== "number") {
        throw new Error("Workload prompt/generation tokens are required");
      }
      const appliedBaseline = filterToAppliedBenchmarkConfig(resolvedConfig.baseline, capabilities);
      const requestedBaselineRecord = tuningSnapshotToRecord(
        resolvedConfig.baseline
      );
      const appliedBaselineRecord = tuningSnapshotToRecord(appliedBaseline);
      const modelRunsPath = path3.join(root, "results", sanitizeFilePart(model.modelName), "runs.jsonl");
      const historicalRecords = await loadRunRecords(modelRunsPath);
      const modelRecords = [];
      for (let repetition = 1; repetition <= resolvedConfig.workload.repetitions; repetition += 1) {
        const runKey = computeRunKey({
          benchCommand: "compare",
          modelPath: model.modelPath,
          appliedConfig: appliedBaseline,
          workload: { pp: promptTokens, tg: generationTokens },
          repetition: { runIndex: repetition, repetitions: resolvedConfig.workload.repetitions },
          hardware: state.hardware,
          runnerIdentity
        });
        const base = {
          id: `run-${now().toISOString().replace(/[:.]/g, "-")}-${repetition}`,
          timestamp: now().toISOString(),
          model: model.modelName,
          run_key: runKey,
          bench_command: "compare",
          runner_identity: runnerIdentity,
          hardware: state.hardware,
          requested_config: requestedBaselineRecord,
          applied_config: appliedBaselineRecord,
          config: appliedBaselineRecord,
          capability_meta: {
            supported_keys: [...capabilities.supportedKeys].sort(),
            used_fallback: capabilities.usedFallback,
            warnings: capabilities.warnings.length > 0 ? [...capabilities.warnings] : void 0
          },
          workload: { pp: promptTokens, tg: generationTokens },
          run_index: repetition,
          repetitions: resolvedConfig.workload.repetitions
        };
        const latest = findLatestRunRecordByRunKey([...historicalRecords, ...modelRecords], runKey);
        const rerun = Boolean(options.rerun);
        const retryFailed = Boolean(options.retryFailed);
        const canReuseLatestSuccess = !rerun && latest?.status === "success";
        if (canReuseLatestSuccess) {
          const record = buildCacheHitRecord(base, latest);
          modelRecords.push(record);
          logger.log(
            `Cache hit: skipping benchmark${retryFailed ? " (--retry-failed)" : ""} (model=${model.modelName}, run_key=${runKey}, reused_run_id=${latest.id})`
          );
          continue;
        }
        try {
          const metrics = await deps.benchRunner.run(
            createBenchInput(
              state,
              model.modelPath,
              appliedBaseline,
              promptTokens,
              generationTokens,
              capabilities
            )
          );
          const record = RunRecordSchema.parse({
            ...base,
            metrics: {
              pp_tokens_per_sec: metrics.ppTokensPerSec,
              tg_tokens_per_sec: metrics.tgTokensPerSec,
              ttft_ms: metrics.ttftMs,
              wall_time_sec: metrics.wallTimeSec ?? null,
              peak_memory_mb: null,
              load_time_ms: null
            },
            status: "success",
            error: null
          });
          modelRecords.push(record);
          await appendRunRecord(modelRunsPath, record);
        } catch (error) {
          hadFailures = true;
          const record = RunRecordSchema.parse({
            ...base,
            metrics: {
              pp_tokens_per_sec: null,
              tg_tokens_per_sec: null,
              ttft_ms: null,
              wall_time_sec: null,
              peak_memory_mb: null,
              load_time_ms: null
            },
            status: "failed",
            error: {
              message: error instanceof Error ? error.message : String(error)
            }
          });
          modelRecords.push(record);
          await appendRunRecord(modelRunsPath, record);
        }
      }
      const summaries = summarizeRecords(modelRecords);
      if (summaries.length === 0) {
        continue;
      }
      const summaryCsv = buildModelSummaryCsv(summaries);
      await writeFile3(path3.join(root, "results", sanitizeFilePart(model.modelName), "summary.csv"), summaryCsv, "utf8");
      const best = summaries[0];
      if (!best) {
        continue;
      }
      rows.push({
        model: model.modelName,
        bestMedianTg: best.medianTg,
        bestMedianPp: best.medianPp,
        bestMedianTtft: best.medianTtft,
        bestConfigKey: best.configKey
      });
    }
    await writeFile3(path3.join(root, "results", "comparison.csv"), buildComparisonCsv(rows), "utf8");
    printModelComparison(logger, rows);
    if (hadFailures) {
      process.exitCode = 1;
    }
  });
}

// src/cli/list-models.ts
import path4 from "path";
function registerListModelsCommand(command, deps) {
  const cwd = deps.cwd ?? (() => process.cwd());
  const logger = deps.logger ?? console;
  command.command("list-models").description("List models discovered by setup").action(async () => {
    const statePath = path4.join(cwd(), ".lmstudio-bench.json");
    if (!await fileExists(statePath)) {
      throw new Error("Run `lmstudio-bench setup` first");
    }
    const state = await readJson(statePath, LocalStateSchema);
    if (state.models.length === 0) {
      logger.log("No models discovered.");
      return;
    }
    for (const model of state.models) {
      logger.log(`${model.name}	${model.path}`);
    }
  });
}

// src/cli/report.ts
import { readdir as readdir2, writeFile as writeFile4 } from "fs/promises";
import path5 from "path";
function registerReportCommand(command, deps) {
  const cwd = deps.cwd ?? (() => process.cwd());
  const logger = deps.logger ?? console;
  command.command("report").description("Regenerate CSV summaries from runs.jsonl files").option("-i, --input <dir>", "Results directory", "results").action(async (options) => {
    const root = path5.resolve(cwd(), options.input ?? "results");
    const entries = await readdir2(root, { withFileTypes: true });
    const modelDirs = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
    const comparisonRows = [];
    let hadWarnings = false;
    for (const modelDir of modelDirs) {
      const runsPath = path5.join(root, modelDir, "runs.jsonl");
      try {
        const { records, invalidLineCount, hadAnyNonEmptyLine } = await loadRunRecordsWithDiagnostics(runsPath);
        if (invalidLineCount > 0 || hadAnyNonEmptyLine && records.length === 0) {
          throw new Error(`runs.jsonl contained ${invalidLineCount} invalid line(s)`);
        }
        const summaries = summarizeRecords(records);
        if (summaries.length === 0) {
          continue;
        }
        const summaryCsv = buildModelSummaryCsv(summaries);
        await writeFile4(path5.join(root, modelDir, "summary.csv"), summaryCsv, "utf8");
        const best = summaries[0];
        if (!best) {
          continue;
        }
        comparisonRows.push({
          model: modelDir,
          bestMedianTg: best.medianTg,
          bestMedianPp: best.medianPp,
          bestMedianTtft: best.medianTtft,
          bestConfigKey: best.configKey
        });
      } catch (error) {
        hadWarnings = true;
        logger.error(
          `Skipping results for ${modelDir}: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }
    const comparisonCsv = buildComparisonCsv(comparisonRows);
    await writeFile4(path5.join(root, "comparison.csv"), comparisonCsv, "utf8");
    logger.log(`Regenerated reports for ${comparisonRows.length} model(s).`);
    if (hadWarnings) {
      process.exitCode = 1;
    }
  });
}

// src/cli/scan.ts
import path6 from "path";
import chalk2 from "chalk";

// src/sweep/sensitivity.ts
function expandSensitivityScan(baseline, scan) {
  const deduped = /* @__PURE__ */ new Map();
  const baselineEntry = { variedParam: "baseline", config: baseline };
  deduped.set(JSON.stringify(baselineEntry.config), baselineEntry);
  const keys = Object.keys(scan);
  for (const key of keys) {
    const values = scan[key] ?? [];
    for (const value of values) {
      const nextConfig = { ...baseline, [key]: value };
      const dedupeKey = JSON.stringify(nextConfig);
      if (!deduped.has(dedupeKey)) {
        deduped.set(dedupeKey, { variedParam: key, config: nextConfig });
      }
    }
  }
  return Array.from(deduped.values());
}

// src/cli/scan.ts
function resolveModelPath2(state, modelRef, config) {
  if (!modelRef) {
    const first = config.models[0];
    if (!first) {
      throw new Error("No models found in sweep config");
    }
    return { modelName: first.name, modelPath: first.path };
  }
  if (modelRef.includes("/") || modelRef.endsWith(".gguf")) {
    return { modelName: path6.basename(modelRef, path6.extname(modelRef)), modelPath: modelRef };
  }
  const fromConfig = config.models.find((item) => item.name === modelRef);
  if (fromConfig) {
    return { modelName: fromConfig.name, modelPath: fromConfig.path };
  }
  const fromState = state.models.find((item) => item.name === modelRef);
  if (fromState) {
    return { modelName: fromState.name, modelPath: fromState.path };
  }
  throw new Error(`Model '${modelRef}' could not be resolved`);
}
function sanitizeFilePart2(value) {
  return value.replace(/[^a-zA-Z0-9._-]/g, "-");
}
function tuningSnapshotToRecord2(source) {
  const out = {};
  for (const [key, value] of Object.entries(source)) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      out[key] = value;
    }
  }
  return out;
}
function buildFailedRecord(base, message) {
  return RunRecordSchema.parse({
    ...base,
    metrics: {
      pp_tokens_per_sec: null,
      tg_tokens_per_sec: null,
      ttft_ms: null,
      peak_memory_mb: null,
      load_time_ms: null,
      wall_time_sec: null
    },
    status: "failed",
    error: {
      message
    }
  });
}
function buildSuccessRecord(base, metrics) {
  return RunRecordSchema.parse({
    ...base,
    metrics: {
      pp_tokens_per_sec: metrics.ppTokensPerSec,
      tg_tokens_per_sec: metrics.tgTokensPerSec,
      ttft_ms: metrics.ttftMs,
      wall_time_sec: metrics.wallTimeSec ?? null,
      peak_memory_mb: null,
      load_time_ms: null
    },
    status: "success",
    error: null
  });
}
function buildCacheHitRecord2(base, reused) {
  return RunRecordSchema.parse({
    ...base,
    metrics: reused.metrics,
    status: "success",
    error: null,
    cache: {
      hit: true,
      reused_run_id: reused.id
    }
  });
}
function registerScanCommand(command, deps) {
  const cwd = deps.cwd ?? (() => process.cwd());
  const now = deps.now ?? (() => /* @__PURE__ */ new Date());
  const logger = deps.logger ?? console;
  command.command("scan").description("Run one-parameter sensitivity scan").option("-m, --model <model>", "Model name or path").option("-c, --config <path>", "Config path", "sweep-config.yaml").option("--estimate-only", "Estimate run count without executing").option("--rerun", "Ignore successful benchmark cache hits and execute all planned runs").option("--retry-failed", "Skip successful cache hits, but rerun failures (based on latest record per run key)").option("-v, --verbose", "Print detailed progress for each run").action(async (options) => {
    const root = cwd();
    const statePath = path6.join(root, ".lmstudio-bench.json");
    if (!await fileExists(statePath)) {
      throw new Error("Run `lmstudio-bench setup` first");
    }
    const configPath = path6.join(root, options.config ?? "sweep-config.yaml");
    const state = await readJson(statePath, LocalStateSchema);
    const capabilities = deps.benchCapabilities ?? await probeLlamaBenchCapabilities(state.llama_bench_path);
    for (const warning of capabilities.warnings) {
      logger.log(`llama-bench capabilities: ${warning}`);
    }
    const config = await readYaml(configPath, SweepConfigSchema);
    const model = resolveModelPath2(state, options.model, config);
    const resolvedConfig = applyModelOverrides(config, {
      name: model.modelName,
      path: model.modelPath
    });
    const unsupportedBaselineKeys = findUnsupportedBaselineKeys(resolvedConfig.baseline, capabilities);
    if (unsupportedBaselineKeys.length > 0) {
      logger.log(
        `Ignoring unsupported baseline fields for this llama-bench build: ${unsupportedBaselineKeys.join(", ")}`
      );
    }
    const unsupportedScanKeys = findUnsupportedScanKeys(resolvedConfig.scan, capabilities);
    if (unsupportedScanKeys.length > 0) {
      logger.log(`Ignoring unsupported scan axes for this llama-bench build: ${unsupportedScanKeys.join(", ")}`);
    }
    const appliedScan = filterScanForCapabilities(resolvedConfig.scan, capabilities);
    const verbose = Boolean(options.verbose);
    if (verbose) {
      logger.log(chalk2.bold("\n\u2500\u2500 Capabilities \u2500\u2500"));
      logger.log(`  Binary: ${state.llama_bench_path}`);
      logger.log(`  Supported keys: ${[...capabilities.supportedKeys].sort().join(", ")}`);
      if (capabilities.usedFallback) {
        logger.log(chalk2.yellow("  (used fallback defaults \u2014 could not probe binary)"));
      }
      logger.log(chalk2.bold("\n\u2500\u2500 Scan Plan \u2500\u2500"));
      const scanAxes = Object.entries(appliedScan).filter(
        ([, v]) => Array.isArray(v) && v.length > 0
      );
      for (const [axis, values] of scanAxes) {
        logger.log(`  ${axis}: [${values.join(", ")}]`);
      }
      logger.log(chalk2.bold("\n\u2500\u2500 Baseline \u2500\u2500"));
      for (const [key, value] of Object.entries(resolvedConfig.baseline)) {
        logger.log(`  ${key}: ${JSON.stringify(value)}`);
      }
    }
    const entries = expandSensitivityScan(resolvedConfig.baseline, appliedScan);
    const runCount = entries.length * resolvedConfig.workload.prompt_tokens.length * resolvedConfig.workload.generation_tokens.length * resolvedConfig.workload.repetitions;
    if (verbose) {
      logger.log(chalk2.bold("\n\u2500\u2500 Run Estimate \u2500\u2500"));
      logger.log(`  ${entries.length} configs \xD7 ${resolvedConfig.workload.prompt_tokens.length} prompt sizes \xD7 ${resolvedConfig.workload.generation_tokens.length} gen sizes \xD7 ${resolvedConfig.workload.repetitions} reps = ${chalk2.cyan(String(runCount))} total runs`);
      logger.log(`  Cache mode: ${options.rerun ? "rerun all (ignoring cache)" : options.retryFailed ? "retry failed only" : "skip cached successes"}`);
      logger.log("");
    }
    if (options.estimateOnly) {
      logger.log(`Estimated runs: ${runCount}`);
      return;
    }
    const runsFilePath = path6.join(root, "results", sanitizeFilePart2(model.modelName), "runs.jsonl");
    const historicalRecords = await loadRunRecords(runsFilePath);
    const runnerIdentity = await deps.benchRunner.getRunnerIdentity?.({ llamaBenchPath: state.llama_bench_path }) ?? await inferLlamaBenchBinaryFingerprint(state.llama_bench_path);
    const records = [];
    let runIndex = 0;
    let hadFailures = false;
    let cacheHits = 0;
    let executed = 0;
    let failed = 0;
    const scanStartTime = Date.now();
    for (const entry of entries) {
      for (const promptTokens of resolvedConfig.workload.prompt_tokens) {
        for (const generationTokens of resolvedConfig.workload.generation_tokens) {
          for (let repetition = 1; repetition <= resolvedConfig.workload.repetitions; repetition += 1) {
            runIndex += 1;
            const appliedRow = filterToAppliedBenchmarkConfig(
              entry.config,
              capabilities
            );
            const runKey = computeRunKey({
              benchCommand: "scan",
              modelPath: model.modelPath,
              appliedConfig: appliedRow,
              workload: { pp: promptTokens, tg: generationTokens },
              repetition: { runIndex: repetition, repetitions: resolvedConfig.workload.repetitions },
              hardware: state.hardware,
              runnerIdentity
            });
            const requestedRecord = tuningSnapshotToRecord2(entry.config);
            const appliedRecord = tuningSnapshotToRecord2(appliedRow);
            const configSummary = Object.entries(appliedRecord).map(([k, v]) => `${k}=${v}`).join(" ");
            const progressPrefix = `[${runIndex}/${runCount}]`;
            const baseRecord = {
              id: `run-${now().toISOString().replace(/[:.]/g, "-")}-${runIndex}`,
              timestamp: now().toISOString(),
              model: model.modelName,
              run_key: runKey,
              bench_command: "scan",
              runner_identity: runnerIdentity,
              hardware: state.hardware,
              requested_config: requestedRecord,
              applied_config: appliedRecord,
              config: appliedRecord,
              capability_meta: {
                supported_keys: [...capabilities.supportedKeys].sort(),
                used_fallback: capabilities.usedFallback,
                warnings: capabilities.warnings.length > 0 ? [...capabilities.warnings] : void 0
              },
              workload: {
                pp: promptTokens,
                tg: generationTokens
              },
              run_index: repetition,
              repetitions: resolvedConfig.workload.repetitions
            };
            const latest = findLatestRunRecordByRunKey([...historicalRecords, ...records], runKey);
            const rerun = Boolean(options.rerun);
            const retryFailed = Boolean(options.retryFailed);
            const canReuseLatestSuccess = !rerun && latest?.status === "success";
            if (canReuseLatestSuccess) {
              cacheHits += 1;
              const record = buildCacheHitRecord2(baseRecord, latest);
              records.push(record);
              if (verbose) {
                logger.log(
                  `${chalk2.dim(progressPrefix)} ${chalk2.yellow("CACHED")} pp=${promptTokens} tg=${generationTokens} | ${configSummary}`
                );
              }
              continue;
            }
            if (verbose) {
              logger.log(
                `${chalk2.cyan(progressPrefix)} ${chalk2.bold("RUNNING")} pp=${promptTokens} tg=${generationTokens} rep=${repetition} | ${configSummary}`
              );
            }
            const runStart = Date.now();
            try {
              const metrics = await deps.benchRunner.run(
                createBenchInput(state, model.modelPath, appliedRow, promptTokens, generationTokens, capabilities)
              );
              executed += 1;
              const elapsed = ((Date.now() - runStart) / 1e3).toFixed(1);
              const record = buildSuccessRecord(baseRecord, metrics);
              records.push(record);
              await appendRunRecord(runsFilePath, record);
              if (verbose) {
                logger.log(
                  `  ${chalk2.green("\u2713")} tg=${metrics.tgTokensPerSec.toFixed(1)} tok/s  pp=${metrics.ppTokensPerSec.toFixed(1)} tok/s  (${elapsed}s)`
                );
              } else {
                logger.log(
                  `[${runIndex}/${runCount}] tg=${metrics.tgTokensPerSec.toFixed(1)} tok/s  pp=${metrics.ppTokensPerSec.toFixed(1)} tok/s | ${configSummary}`
                );
              }
            } catch (error) {
              hadFailures = true;
              failed += 1;
              const elapsed = ((Date.now() - runStart) / 1e3).toFixed(1);
              const msg = error instanceof Error ? error.message : String(error);
              const record = buildFailedRecord(baseRecord, msg);
              records.push(record);
              await appendRunRecord(runsFilePath, record);
              if (verbose) {
                logger.log(`  ${chalk2.red("\u2717")} FAILED (${elapsed}s): ${msg}`);
              } else {
                logger.log(`[${runIndex}/${runCount}] FAILED | ${configSummary}: ${msg}`);
              }
            }
          }
        }
      }
    }
    if (verbose) {
      const totalElapsed = ((Date.now() - scanStartTime) / 1e3).toFixed(0);
      logger.log(chalk2.bold("\n\u2500\u2500 Scan Complete \u2500\u2500"));
      logger.log(`  Executed: ${executed}  Cached: ${cacheHits}  Failed: ${failed}  Total: ${runCount}`);
      logger.log(`  Wall time: ${totalElapsed}s`);
    }
    const summaries = summarizeRecords(records);
    printScanSummary(logger, summaries, { modelName: model.modelName });
    if (hadFailures) {
      process.exitCode = 1;
    }
  });
}

// src/cli/setup.ts
import path7 from "path";
function createDefaultSweepConfig(state, capabilities) {
  const models = state.models.length > 0 ? state.models : [{ name: "model", path: "<path-to-model.gguf>", size_gb: 0 }];
  const s = capabilities.supportedKeys;
  return SweepConfigSchema.parse({
    models: models.map((model) => ({ path: model.path, name: model.name })),
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
      mlock: false
    },
    scan: {
      n_ctx: s.has("n_ctx") ? [2048, 4096, 8192] : [],
      n_batch: s.has("n_batch") ? [128, 256, 512, 1024] : [],
      n_ubatch: s.has("n_ubatch") ? [64, 128, 256, 512] : [],
      n_gpu_layers: s.has("n_gpu_layers") ? [0, 8, 16, 24, 30, 99] : [],
      threads: s.has("threads") ? [0, 4, 6, 8, 10, 12] : [],
      threads_batch: s.has("threads_batch") ? [0, 4, 6, 8, 10, 12] : [],
      kv_type_key: s.has("kv_type_key") ? ["f16", "f32", "s16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] : [],
      kv_type_value: s.has("kv_type_value") ? ["f16", "f32", "s16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] : [],
      flash_attention: s.has("flash_attention") ? [true, false] : [],
      mmap: s.has("mmap") ? [true, false] : [],
      mlock: s.has("mlock") ? [false, true] : []
    },
    workload: {
      prompt_tokens: [512],
      generation_tokens: [128],
      repetitions: 3
    },
    model_overrides: []
  });
}
function registerSetupCommand(command, deps) {
  const cwd = deps.cwd ?? (() => process.cwd());
  const logger = deps.logger ?? console;
  command.command("setup").description("Discover llama.cpp tools and local models").action(async () => {
    const [tools, hardware, models] = await Promise.all([
      deps.discoveryService.discoverTools(),
      deps.discoveryService.discoverHardware(),
      deps.discoveryService.discoverModels()
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
        metal: hardware.metal
      },
      model_dirs: models.modelDirs,
      models: models.models.map((model) => ({
        name: model.name,
        path: model.path,
        size_gb: model.sizeGb
      }))
    });
    const statePath = path7.join(cwd(), ".lmstudio-bench.json");
    const sweepConfigPath = path7.join(cwd(), "sweep-config.yaml");
    await writeJson(statePath, state);
    const capabilities = deps.benchCapabilities ?? await probeLlamaBenchCapabilities(state.llama_bench_path);
    const wroteConfig = await writeYamlIfMissing(sweepConfigPath, createDefaultSweepConfig(state, capabilities));
    logger.log(`Setup complete. Found ${state.models.length} model(s).`);
    if (wroteConfig) {
      logger.log("Created sweep-config.yaml starter config.");
    }
  });
}

// src/cli/index.ts
var CLI_NAME = "lmstudio-bench";
function createCli(deps = {}) {
  const command = new Command().name(CLI_NAME).description("Benchmark llama.cpp configurations");
  const logger = deps.logger ?? console;
  const benchRunner = deps.benchRunner ?? createBenchRunner();
  registerSetupCommand(command, {
    discoveryService: deps.discoveryService ?? createDiscoveryService(),
    ...deps.benchCapabilities !== void 0 ? { benchCapabilities: deps.benchCapabilities } : {},
    logger
  });
  registerListModelsCommand(command, {
    logger
  });
  registerScanCommand(command, {
    benchRunner,
    ...deps.benchCapabilities !== void 0 ? { benchCapabilities: deps.benchCapabilities } : {},
    logger
  });
  registerCompareCommand(command, {
    benchRunner,
    ...deps.benchCapabilities !== void 0 ? { benchCapabilities: deps.benchCapabilities } : {},
    logger
  });
  registerReportCommand(command, {
    logger
  });
  return command;
}

// src/cli.ts
async function main() {
  const cli = createCli();
  await cli.parseAsync(process.argv);
}
void main();
//# sourceMappingURL=cli.js.map