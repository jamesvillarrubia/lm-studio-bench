import os from "os";
import path from "path";
import { mkdir } from "fs/promises";

const DEFAULT_DIR_NAME = ".lmstudio-bench";

/**
 * Resolved data root directory. All config, results, and cache live here.
 *
 * Priority:
 *   1. --data-dir CLI flag
 *   2. LMSTUDIO_BENCH_HOME env var
 *   3. ~/.lmstudio-bench/
 */
export function resolveDataRoot(override?: string): string {
  if (override) return path.resolve(override);
  const env = process.env["LMSTUDIO_BENCH_HOME"];
  if (env) return path.resolve(env);
  return path.join(os.homedir(), DEFAULT_DIR_NAME);
}

export async function ensureDataRoot(root: string): Promise<void> {
  await mkdir(root, { recursive: true });
}

export function statePath(root: string): string {
  return path.join(root, "config.json");
}

export function sweepConfigPath(
  root: string,
  override?: string,
): string {
  if (override) return path.resolve(override);
  return path.join(root, "sweep-config.yaml");
}

export function resultsDir(root: string): string {
  return path.join(root, "results");
}

export function modelRunsPath(
  root: string,
  modelDirName: string,
): string {
  return path.join(root, "results", modelDirName, "runs.jsonl");
}

export function modelSummaryPath(
  root: string,
  modelDirName: string,
): string {
  return path.join(root, "results", modelDirName, "summary.csv");
}

export function comparisonCsvPath(root: string): string {
  return path.join(root, "results", "comparison.csv");
}
