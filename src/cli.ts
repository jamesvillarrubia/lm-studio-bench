import { createCli } from "./cli/index.js";

async function main(): Promise<void> {
  const cli = createCli();
  await cli.parseAsync(process.argv);
}

void main();
