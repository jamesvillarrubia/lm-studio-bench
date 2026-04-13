import { spawn } from "child_process";

export interface SubprocessResult {
  stdout: string;
  stderr: string;
  exitCode: number;
  elapsedMs: number;
}

export async function runSubprocess(
  command: string,
  args: readonly string[],
): Promise<SubprocessResult> {
  const start = Date.now();
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer | string) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk: Buffer | string) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error: Error) => {
      reject(error);
    });
    child.on("close", (exitCode: number | null) => {
      resolve({
        stdout,
        stderr,
        exitCode: exitCode ?? 1,
        elapsedMs: Date.now() - start,
      });
    });
  });
}
