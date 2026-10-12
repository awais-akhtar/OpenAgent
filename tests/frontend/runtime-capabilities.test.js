const { test } = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { buildSync } = require("esbuild");

const bundle = buildSync({
  entryPoints: [path.resolve(__dirname, "../../web/src/runtime-capabilities.ts")],
  bundle: true,
  platform: "node",
  format: "cjs",
  write: false,
  logLevel: "silent",
}).outputFiles[0].text;
const moduleShim = { exports: {} };
new Function("module", "exports", bundle)(moduleShim, moduleShim.exports);
const { normalizeRuntimeCapabilities } = moduleShim.exports;

test("runtime capability states do not equate a responding probe with readiness", () => {
  const capabilities = normalizeRuntimeCapabilities({ modules: [
    { id: "chat", probe_status: "responding", capability_status: "setup_needed", reason: "Install a local model." },
    { id: "science", probe_status: "responding", capability_status: "not_evaluated", evidence: { status: "mock-completed" } },
    { id: "voice", probe_status: "error", capability_status: "not_evaluated" },
    { id: "models", probe_status: "responding", capability_status: "available" },
  ] });
  assert.deepEqual(capabilities.map(item => [item.id, item.state]), [
    ["chat", "setup-needed"],
    ["science", "sandbox"],
    ["voice", "degraded"],
    ["models", "ready"],
  ]);
  assert.equal(capabilities[0].reason, "Install a local model.");
});
