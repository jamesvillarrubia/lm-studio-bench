import { access, readFile, writeFile } from "fs/promises";
import { constants } from "fs";
import { parse, stringify } from "yaml";
import type { z } from "zod";

export async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export async function writeJson(
  filePath: string,
  value: unknown,
): Promise<void> {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export async function readJson<T>(
  filePath: string,
  schema: z.ZodType<T>,
): Promise<T> {
  const raw = await readFile(filePath, "utf8");
  const parsed = JSON.parse(raw);
  return schema.parse(parsed);
}

export async function writeYaml(
  filePath: string,
  value: unknown,
): Promise<void> {
  await writeFile(filePath, stringify(value), "utf8");
}

export async function readYaml<T>(
  filePath: string,
  schema: z.ZodType<T>,
): Promise<T> {
  const raw = await readFile(filePath, "utf8");
  const parsed = parse(raw);
  return schema.parse(parsed);
}

export async function writeYamlIfMissing(
  filePath: string,
  value: unknown,
): Promise<boolean> {
  if (await fileExists(filePath)) {
    return false;
  }
  await writeYaml(filePath, value);
  return true;
}
