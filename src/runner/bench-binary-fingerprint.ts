import { createHash } from "crypto";
import { open, stat } from "fs/promises";

const HEAD_BYTES = 1024 * 1024;

export async function inferLlamaBenchBinaryFingerprint(
  llamaBenchPath: string,
): Promise<string> {
  try {
    const fileStats = await stat(llamaBenchPath);
    const file = await open(llamaBenchPath, "r");
    try {
      const buffer = Buffer.allocUnsafe(
        Math.min(HEAD_BYTES, Number(fileStats.size)),
      );
      if (buffer.length === 0) {
        return `llama_bench_binary|path:${llamaBenchPath}|size:0|mtime_ms:${Number(fileStats.mtimeMs)}|head_sha256:empty`;
      }
      const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
      const head = buffer.subarray(0, bytesRead);
      const headSha256 = createHash("sha256").update(head).digest("hex");
      return `llama_bench_binary|path:${llamaBenchPath}|size:${fileStats.size}|mtime_ms:${Number(
        fileStats.mtimeMs,
      )}|head_sha256:${headSha256}`;
    } finally {
      await file.close();
    }
  } catch {
    return `llama_bench_binary|missing_path:${llamaBenchPath}`;
  }
}
