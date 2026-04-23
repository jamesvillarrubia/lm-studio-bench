import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import { RunRecordSchema } from "../config/schema.js";
import type { RunRecord } from "../config/schema.js";

export interface ConfigSummary {
  configKey: string;
  config: Record<string, unknown>;
  successCount: number;
  failedCount: number;
  medianPp: number | null;
  medianTg: number | null;
  medianTtft: number | null;
}

function median(values: number[]): number | null {
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

function recordAppliedConfig(
  record: RunRecord,
): Record<string, string | number | boolean> | undefined {
  return record.applied_config ?? record.config;
}

function configKey(
  config: Record<string, string | number | boolean> | undefined,
): string {
  if (!config) return "{}";
  const sorted = Object.keys(config)
    .sort()
    .map((key) => [key, config[key]]);
  return JSON.stringify(sorted);
}

export async function appendRunRecord(
  runsFilePath: string,
  runRecord: RunRecord,
): Promise<void> {
  await mkdir(path.dirname(runsFilePath), { recursive: true });
  const parsed = RunRecordSchema.parse(runRecord);
  await writeFile(runsFilePath, `${JSON.stringify(parsed)}\n`, {
    encoding: "utf8",
    flag: "a",
  });
}

export function parseRunRecordsFromJsonl(raw: string): RunRecord[] {
  const records: RunRecord[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    let json: unknown;
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

interface RunRecordDiagnostics {
  records: RunRecord[];
  hadAnyNonEmptyLine: boolean;
  invalidLineCount: number;
}

function parseRunRecordsFromJsonlWithDiagnostics(
  raw: string,
): RunRecordDiagnostics {
  const records: RunRecord[] = [];
  let invalidLineCount = 0;
  let hadAnyNonEmptyLine = false;
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    hadAnyNonEmptyLine = true;
    let json: unknown;
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

export async function loadRunRecords(
  runsFilePath: string,
): Promise<RunRecord[]> {
  try {
    const raw = await readFile(runsFilePath, "utf8");
    return parseRunRecordsFromJsonl(raw);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

export async function loadRunRecordsWithDiagnostics(
  runsFilePath: string,
): Promise<RunRecordDiagnostics> {
  try {
    const raw = await readFile(runsFilePath, "utf8");
    return parseRunRecordsFromJsonlWithDiagnostics(raw);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      return { records: [], hadAnyNonEmptyLine: false, invalidLineCount: 0 };
    }
    throw error;
  }
}

export function findLatestRunRecordByRunKey(
  records: RunRecord[],
  runKey: string,
): RunRecord | null {
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

export interface ExecutionSignature {
  benchCommand: "scan" | "compare";
  appliedConfig: Record<string, string | number | boolean>;
  workload: { pp: number; tg: number };
  runIndex: number;
  repetitions: number;
}

function hasOwn<T extends object>(obj: T, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

function isCompatibleAppliedConfig(
  expected: Record<string, string | number | boolean>,
  candidate:
    | Record<string, string | number | boolean>
    | undefined,
): boolean {
  if (!candidate) {
    return false;
  }
  // Backward-compatibility: older runs may omit newly-added boolean flags.
  // Treat missing keys as equivalent only when expected value is false.
  for (const [key, expectedValue] of Object.entries(expected)) {
    if (!hasOwn(candidate, key)) {
      if (typeof expectedValue === "boolean" && expectedValue === false) {
        continue;
      }
      return false;
    }
    if (candidate[key] !== expectedValue) {
      return false;
    }
  }
  return true;
}

/**
 * Legacy-compatible cache lookup when runner_identity changes.
 * Matches by execution semantics (bench command + applied config + workload + run index),
 * ignoring run_key, runner_identity, and total repetition count.
 *
 * We intentionally do not require `repetitions` to match because rep #1 for a
 * config/workload is still valid when the user changes the planned total from
 * 3 -> 5 or 5 -> 3.
 */
export function findLatestRunRecordByExecutionSignature(
  records: RunRecord[],
  signature: ExecutionSignature,
): RunRecord | null {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const record = records[index];
    if (!record) {
      continue;
    }
    if (record.bench_command && record.bench_command !== signature.benchCommand) {
      continue;
    }
    if (record.workload.pp !== signature.workload.pp || record.workload.tg !== signature.workload.tg) {
      continue;
    }
    if (record.run_index !== signature.runIndex) {
      continue;
    }
    if (!isCompatibleAppliedConfig(signature.appliedConfig, recordAppliedConfig(record))) {
      continue;
    }
    return record;
  }
  return null;
}

export function summarizeRecords(records: RunRecord[]): ConfigSummary[] {
  const byConfig = new Map<string, RunRecord[]>();
  for (const record of records) {
    const applied = recordAppliedConfig(record);
    const key = configKey(applied);
    const existing = byConfig.get(key) ?? [];
    existing.push(record);
    byConfig.set(key, existing);
  }
  const summaries: ConfigSummary[] = [];
  for (const [key, grouped] of byConfig.entries()) {
    const successes = grouped.filter((item) => item.status === "success");
    const failures = grouped.filter((item) => item.status === "failed");
    const ppValues = successes
      .map((item) => item.metrics.pp_tokens_per_sec)
      .filter((value): value is number => typeof value === "number");
    const tgValues = successes
      .map((item) => item.metrics.tg_tokens_per_sec)
      .filter((value): value is number => typeof value === "number");
    const ttftValues = successes
      .map((item) => item.metrics.ttft_ms)
      .filter((value): value is number => typeof value === "number");
    summaries.push({
      configKey: key,
      config: grouped[0] ? (recordAppliedConfig(grouped[0]) ?? {}) : {},
      successCount: successes.length,
      failedCount: failures.length,
      medianPp: median(ppValues),
      medianTg: median(tgValues),
      medianTtft: median(ttftValues),
    });
  }
  return summaries.sort(
    (left, right) => (right.medianTg ?? -1) - (left.medianTg ?? -1),
  );
}
