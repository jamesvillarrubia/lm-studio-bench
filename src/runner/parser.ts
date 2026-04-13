export interface BenchMetrics {
  ppTokensPerSec: number;
  tgTokensPerSec: number;
  ttftMs: number | null;
}

interface LegacyJsonObject {
  pp_tokens_per_sec?: unknown;
  tg_tokens_per_sec?: unknown;
  ttft_ms?: unknown;
}

interface ModernJsonEntry {
  n_prompt?: unknown;
  n_gen?: unknown;
  avg_ts?: unknown;
}

function parseLegacyJsonObject(parsed: LegacyJsonObject): BenchMetrics | null {
  const pp = Number(parsed.pp_tokens_per_sec);
  const tg = Number(parsed.tg_tokens_per_sec);
  const ttftValue = Number(parsed.ttft_ms);
  const ttft = Number.isFinite(ttftValue) ? ttftValue : null;
  if ([pp, tg].every(Number.isFinite)) {
    return {
      ppTokensPerSec: pp,
      tgTokensPerSec: tg,
      ttftMs: ttft,
    };
  }
  return null;
}

function parseModernJsonArray(parsed: ModernJsonEntry[]): BenchMetrics | null {
  const promptEntry = parsed.find(
    (entry) =>
      typeof entry === "object" &&
      entry !== null &&
      Number(entry.n_prompt) > 0 &&
      Number(entry.n_gen) === 0,
  );
  const generationEntry = parsed.find(
    (entry) =>
      typeof entry === "object" &&
      entry !== null &&
      Number(entry.n_gen) > 0 &&
      Number(entry.n_prompt) === 0,
  );
  const pp = Number(promptEntry?.avg_ts);
  const tg = Number(generationEntry?.avg_ts);
  if ([pp, tg].every(Number.isFinite)) {
    return {
      ppTokensPerSec: pp,
      tgTokensPerSec: tg,
      ttftMs: null,
    };
  }
  return null;
}

function parseJsonOutput(raw: string): BenchMetrics | null {
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

function parseMetricByRegex(
  raw: string,
  patterns: RegExp[],
): number | null {
  for (const pattern of patterns) {
    const match = raw.match(pattern);
    const value = Number(match?.[1] ?? Number.NaN);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

export function parseLlamaBenchOutput(raw: string): BenchMetrics {
  const fromJson = parseJsonOutput(raw);
  if (fromJson) {
    return fromJson;
  }
  const pp = parseMetricByRegex(raw, [
    /pp(?:_tokens_per_sec)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
    /prompt(?:\s+processing)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
  ]);
  const tg = parseMetricByRegex(raw, [
    /tg(?:_tokens_per_sec)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
    /generation(?:\s+speed)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
  ]);
  const ttft = parseMetricByRegex(raw, [
    /ttft(?:_ms)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
    /time\s+to\s+first\s+token\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)/i,
  ]);
  if (pp === null || tg === null) {
    throw new Error("Unable to parse llama-bench output");
  }
  return {
    ppTokensPerSec: pp,
    tgTokensPerSec: tg,
    ttftMs: ttft ?? null,
  };
}
