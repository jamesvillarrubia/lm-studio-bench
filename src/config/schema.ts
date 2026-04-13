import { z } from "zod";

export const LocalModelSchema = z.object({
  name: z.string().min(1),
  path: z.string().min(1),
  size_gb: z.number().nonnegative(),
});

export const LocalStateSchema = z.object({
  llama_bench_path: z.string().min(1),
  llama_cli_path: z.string().min(1),
  hardware: z.object({
    chip: z.string().min(1),
    cores: z.number().int().positive(),
    p_cores: z.number().int().nonnegative(),
    e_cores: z.number().int().nonnegative(),
    memory_gb: z.number().int().positive(),
    metal: z.boolean(),
  }),
  model_dirs: z.array(z.string().min(1)).default([]),
  models: z.array(LocalModelSchema).default([]),
});

export type LocalState = z.infer<typeof LocalStateSchema>;

export const SweepModelSchema = z.object({
  path: z.string().min(1),
  name: z.string().min(1),
});

export const KvCacheTypeSchema = z.string().min(1);

export const BenchmarkBaselineSchema = z.object({
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
  mlock: z.boolean(),
  no_kv_offload: z.boolean(),
  n_cpu_moe: z.number().int().nonnegative(),
});

export type BenchmarkBaseline = z.infer<typeof BenchmarkBaselineSchema>;

export const BenchmarkScanSchema = z.object({
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
  mlock: z.array(z.boolean()).default([]),
  no_kv_offload: z.array(z.boolean()).default([]),
  n_cpu_moe: z.array(z.number().int().nonnegative()).default([]),
});

export type BenchmarkScan = z.infer<typeof BenchmarkScanSchema>;

export const WorkloadSchema = z.object({
  prompt_tokens: z.array(z.number().int().positive()).default([512]),
  generation_tokens: z.array(z.number().int().positive()).default([128]),
  repetitions: z.number().int().positive().default(3),
});

export type Workload = z.infer<typeof WorkloadSchema>;

export const AxisHintSchema = z.object({
  direction: z.enum([
    "max",
    "min",
    "prefer_true",
    "prefer_false",
    "unconstrained",
  ]),
  reason: z.string().optional(),
});

export const SweepConfigSchema = z.object({
  models: z.array(SweepModelSchema).min(1),
  baseline: BenchmarkBaselineSchema,
  scan: BenchmarkScanSchema,
  workload: WorkloadSchema,
  axis_hints: z.record(z.string(), AxisHintSchema).default({}),
  model_overrides: z
    .array(
      z.object({
        match: z
          .object({
            name: z.string().min(1).optional(),
            path: z.string().min(1).optional(),
          })
          .refine(
            (value) =>
              typeof value.name === "string" ||
              typeof value.path === "string",
            {
              message:
                "model_overrides.match requires at least one of name or path",
            },
          ),
        baseline: BenchmarkBaselineSchema.partial().optional(),
        scan: BenchmarkScanSchema.partial().optional(),
        workload: WorkloadSchema.partial().optional(),
      }),
    )
    .default([]),
});

export type SweepConfig = z.infer<typeof SweepConfigSchema>;

const runRecordScalar = z.union([z.string(), z.number(), z.boolean()]);

export const RunRecordSchema = z
  .object({
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
    cache: z
      .object({
        /**
         * When true, this record was synthesized from a prior successful JSONL record and
         * did not execute a fresh benchmark in this session.
         */
        hit: z.boolean(),
        /**
         * The `id` of the prior record this cache hit reused (best-effort; omitted for legacy hits).
         */
        reused_run_id: z.string().min(1).optional(),
      })
      .optional(),
    /**
     * YAML / sweep row as interpreted for this run (may include fields not passed to `llama-bench`).
     */
    requested_config: z.record(z.string(), runRecordScalar).optional(),
    /**
     * Subset of `requested_config` that was actually passed to `llama-bench` for this run.
     */
    applied_config: z.record(z.string(), runRecordScalar).optional(),
    capability_meta: z
      .object({
        supported_keys: z.array(z.string()),
        used_fallback: z.boolean(),
        warnings: z.array(z.string()).optional(),
      })
      .optional(),
    /** @deprecated Prefer `applied_config`; retained for older JSONL rows. */
    config: z.record(z.string(), runRecordScalar).optional(),
    workload: z.object({
      pp: z.number().int().positive(),
      tg: z.number().int().positive(),
    }),
    metrics: z.object({
      pp_tokens_per_sec: z.number().nullable(),
      tg_tokens_per_sec: z.number().nullable(),
      ttft_ms: z.number().nullable(),
      peak_memory_mb: z.number().nullable().optional(),
      load_time_ms: z.number().nullable().optional(),
      wall_time_sec: z.number().nullable().optional(),
    }),
    run_index: z.number().int().positive(),
    repetitions: z.number().int().positive(),
    status: z.enum(["success", "failed"]),
    error: z
      .object({
        message: z.string().min(1),
        code: z.string().optional(),
      })
      .nullable(),
  })
  .superRefine((value, ctx) => {
    if (!value.config && !value.applied_config) {
      ctx.addIssue({
        code: "custom",
        message: "RunRecord requires config (legacy) or applied_config",
        path: ["applied_config"],
      });
    }
  });

export type RunRecord = z.infer<typeof RunRecordSchema>;
