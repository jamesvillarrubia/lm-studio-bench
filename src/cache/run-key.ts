import { createHash } from "crypto";

type JsonPrimitive = string | number | boolean | null;
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

function stableStringify(value: JsonValue): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  const entries = Object.entries(value)
    .filter(([, v]) => typeof v !== "undefined")
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(",")}}`;
}

export interface RunKeyInput {
  benchCommand: string;
  modelPath: string;
  appliedConfig: Record<string, string | number | boolean>;
  workload: { pp: number; tg: number };
  repetition: { runIndex: number; repetitions: number };
  hardware: Record<string, unknown>;
  runnerIdentity: string;
}

export function computeRunKey(input: RunKeyInput): string {
  const payload = stableStringify({
    v: 1,
    benchCommand: input.benchCommand,
    modelPath: input.modelPath,
    appliedConfig: input.appliedConfig as unknown as JsonValue,
    workload: input.workload as unknown as JsonValue,
    repetition: input.repetition as unknown as JsonValue,
    hardware: input.hardware as unknown as JsonValue,
    runnerIdentity: input.runnerIdentity,
  });
  return createHash("sha256").update(payload, "utf8").digest("hex");
}
