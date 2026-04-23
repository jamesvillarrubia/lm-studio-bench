/**
 * Axis behavior hints that let the adaptive strategy bias sampling toward
 * values that are likely to be good without removing coverage entirely.
 *
 * - `max`: Higher numeric value is always better (e.g. n_gpu_layers on unified memory).
 *          Pin to the maximum value in the scan range.
 * - `min`: Lower numeric value is always better. Pin to the minimum.
 * - `prefer_true`: Boolean axis where true always wins (e.g. flash_attention).
 * - `prefer_false`: Boolean axis where false always wins (e.g. no_kv_offload).
 * - `unconstrained`: No known monotonic relationship — must be swept.
 */
export type AxisHintDirection =
  | "max"
  | "min"
  | "prefer_true"
  | "prefer_false"
  | "unconstrained";

export interface AxisHint {
  direction: AxisHintDirection;
  /** Human-readable reason (printed in verbose mode). */
  reason: string;
}

export type AxisHintMap = Record<string, AxisHint>;

/**
 * Default hints for known axes. These encode widely-understood hardware
 * behavior and can be overridden per-config via `axis_hints` in YAML.
 */
export function getDefaultAxisHints(hardware: {
  metal: boolean;
  memory_gb: number;
}): AxisHintMap {
  const hints: AxisHintMap = {};

  if (hardware.metal) {
    // Apple Silicon unified memory: max GPU offload is always optimal
    hints.n_gpu_layers = {
      direction: "max",
      reason: "Unified memory — full GPU offload is always fastest",
    };
    hints.no_kv_offload = {
      direction: "prefer_false",
      reason: "Unified memory — KV on GPU avoids CPU-GPU copies",
    };
  }

  // flash_attention is universally better when supported
  hints.flash_attention = {
    direction: "prefer_true",
    reason: "Flash attention reduces memory and improves cache locality",
  };

  if (hardware.memory_gb >= 32) {
    hints.mmap = {
      direction: "prefer_true",
      reason: "Sufficient RAM — mmap avoids redundant copies",
    };
  }

  // n_cpu_moe=0 means "auto" in llama.cpp; on non-MoE models all values are equivalent.
  // Not pinnable in general since MoE models genuinely vary.

  return hints;
}

/**
 * Given hints, determine the optimal pinned value for an axis (if known),
 * or return null if the axis needs sweeping.
 */
export function getPinnedValue(
  axis: string,
  scanValues: (string | number | boolean)[],
  hint: AxisHint,
): string | number | boolean | null {
  if (scanValues.length === 0) return null;

  switch (hint.direction) {
    case "max": {
      const nums = scanValues.filter((v): v is number => typeof v === "number");
      return nums.length > 0 ? Math.max(...nums) : null;
    }
    case "min": {
      const nums = scanValues.filter((v): v is number => typeof v === "number");
      return nums.length > 0 ? Math.min(...nums) : null;
    }
    case "prefer_true":
      return scanValues.includes(true) ? true : null;
    case "prefer_false":
      return scanValues.includes(false) ? false : null;
    case "unconstrained":
      return null;
  }
}

/**
 * Apply hints as soft preferences.
 * Returns preferred overrides to bias the baseline/LHS sampling, while keeping
 * every sweep axis active so all levels still receive coverage.
 */
export function applyAxisHints(
  scan: Record<string, (string | number | boolean)[]>,
  hints: AxisHintMap,
): {
  pinnedOverrides: Record<string, string | number | boolean>;
  reducedScan: Record<string, (string | number | boolean)[]>;
  pinLog: { axis: string; value: string | number | boolean; reason: string }[];
} {
  const pinnedOverrides: Record<string, string | number | boolean> = {};
  const reducedScan: Record<string, (string | number | boolean)[]> = {};
  const pinLog: {
    axis: string;
    value: string | number | boolean;
    reason: string;
  }[] = [];

  for (const [axis, values] of Object.entries(scan)) {
    const hint = hints[axis];
    if (!hint || hint.direction === "unconstrained") {
      reducedScan[axis] = values;
      continue;
    }

    const pinned = getPinnedValue(axis, values, hint);
    if (pinned !== null) {
      pinnedOverrides[axis] = pinned;
      pinLog.push({ axis, value: pinned, reason: hint.reason });
      reducedScan[axis] = values;
    } else {
      reducedScan[axis] = values;
    }
  }

  return { pinnedOverrides, reducedScan, pinLog };
}
