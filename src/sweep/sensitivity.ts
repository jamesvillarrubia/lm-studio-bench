export interface BenchmarkConfig {
  [key: string]: string | number | boolean;
}

export interface SensitivityEntry {
  variedParam: string;
  config: BenchmarkConfig;
}

export function expandSensitivityScan(
  baseline: BenchmarkConfig,
  scan: Record<string, (string | number | boolean)[]>,
): SensitivityEntry[] {
  const deduped = new Map<string, SensitivityEntry>();
  const baselineEntry: SensitivityEntry = {
    variedParam: "baseline",
    config: baseline,
  };
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
