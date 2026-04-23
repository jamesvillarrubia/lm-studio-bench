/**
 * Standalone CLI via `bun build --compile`.
 * Local: writes artifacts/lmstudio-bench (or .exe on Windows).
 * CI: set BUN_COMPILE_TARGET (e.g. bun-linux-x64) for cross-compilation.
 */
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = process.env.BUN_COMPILE_TARGET?.trim() ?? "";
const outDir = join(root, "artifacts");
await mkdir(outDir, { recursive: true });

const windows =
  target.includes("windows") ||
  (process.platform === "win32" && !target);
const outfile = join(outDir, windows ? "lmstudio-bench.exe" : "lmstudio-bench");

const cmd = [
  "bun",
  "build",
  "--compile",
  join(root, "src", "cli.ts"),
  "--outfile",
  outfile
];
if (target) cmd.push("--target", target);

const proc = Bun.spawnSync({
  cmd,
  cwd: root,
  stdout: "inherit",
  stderr: "inherit"
});

process.exit(proc.exitCode ?? 1);
