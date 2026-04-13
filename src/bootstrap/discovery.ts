import { access, readdir, stat } from "fs/promises";
import { constants } from "fs";
import os from "os";
import path from "path";

export const DEFAULT_MODEL_DIRS = [
  "~/.lmstudio/models",
  "~/Library/Application Support/LM Studio/models",
] as const;

export interface DiscoveredTools {
  llamaBenchPath: string;
  llamaCliPath: string;
}

export interface DiscoveredHardware {
  chip: string;
  cores: number;
  pCores: number;
  eCores: number;
  memoryGb: number;
  metal: boolean;
}

export interface DiscoveredModel {
  name: string;
  path: string;
  sizeGb: number;
}

export interface DiscoveredModels {
  modelDirs: readonly string[];
  models: DiscoveredModel[];
}

export interface DiscoveryService {
  discoverTools(): Promise<DiscoveredTools>;
  discoverHardware(): Promise<DiscoveredHardware>;
  discoverModels(): Promise<DiscoveredModels>;
}

function expandHome(dir: string): string {
  if (!dir.startsWith("~/")) {
    return dir;
  }
  return path.join(os.homedir(), dir.slice(2));
}

async function isExecutable(filePath: string): Promise<boolean> {
  try {
    await access(filePath, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findOnPath(binaryName: string): Promise<string | null> {
  const envPath = process.env.PATH ?? "";
  const segments = envPath.split(path.delimiter).filter(Boolean);
  for (const segment of segments) {
    const candidate = path.join(segment, binaryName);
    if (await isExecutable(candidate)) {
      return candidate;
    }
  }
  return null;
}

function isRunnableModelFile(fileName: string): boolean {
  if (!fileName.endsWith(".gguf")) {
    return false;
  }
  const lower = fileName.toLowerCase();
  if (
    lower.startsWith("mmproj-") ||
    lower.includes("-mmproj-") ||
    lower.includes("mmproj")
  ) {
    return false;
  }
  return true;
}

async function walkForGgufFiles(root: string): Promise<DiscoveredModel[]> {
  const expandedRoot = expandHome(root);
  try {
    const entries = await readdir(expandedRoot, {
      withFileTypes: true,
      recursive: true,
    });
    const models: DiscoveredModel[] = [];
    for (const entry of entries) {
      if (!entry.isFile() || !isRunnableModelFile(entry.name)) {
        continue;
      }
      const fullPath = path.join(entry.parentPath, entry.name);
      const fileStats = await stat(fullPath);
      models.push({
        name: entry.name.replace(/\.gguf$/i, ""),
        path: fullPath,
        sizeGb: Number((fileStats.size / 1024 / 1024 / 1024).toFixed(2)),
      });
    }
    return models;
  } catch {
    return [];
  }
}

export function createDiscoveryService(): DiscoveryService {
  return {
    async discoverTools(): Promise<DiscoveredTools> {
      const llamaBenchPath = await findOnPath("llama-bench");
      const llamaCliPath = await findOnPath("llama-cli");
      if (!llamaBenchPath || !llamaCliPath) {
        throw new Error("Could not find llama-bench and llama-cli on PATH");
      }
      return { llamaBenchPath, llamaCliPath };
    },
    async discoverHardware(): Promise<DiscoveredHardware> {
      const cores = os.cpus().length;
      return {
        chip: process.platform === "darwin" ? "Apple Silicon" : process.arch,
        cores,
        pCores: Math.max(1, Math.ceil(cores * 0.66)),
        eCores: Math.max(0, cores - Math.ceil(cores * 0.66)),
        memoryGb: Math.round(os.totalmem() / 1024 / 1024 / 1024),
        metal: process.platform === "darwin",
      };
    },
    async discoverModels(): Promise<DiscoveredModels> {
      const modelBatches = await Promise.all(
        DEFAULT_MODEL_DIRS.map((dir) => walkForGgufFiles(dir)),
      );
      return {
        modelDirs: DEFAULT_MODEL_DIRS,
        models: modelBatches.flat(),
      };
    },
  };
}
