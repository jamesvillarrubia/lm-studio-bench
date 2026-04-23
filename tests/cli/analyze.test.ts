import { describe, expect, test } from "vitest";
import { createCli } from "../../src/cli/index.js";

describe("CLI", () => {
  test("registers the analyze command", () => {
    const cli = createCli();
    const commandNames = cli.commands.map((command) => command.name());
    expect(commandNames).toContain("analyze");
  });
});
