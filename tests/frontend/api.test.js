const { test } = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { buildSync } = require("esbuild");

const bundle = buildSync({
  entryPoints: [path.resolve(__dirname, "../../web/src/api.ts")],
  bundle: true,
  platform: "node",
  format: "cjs",
  write: false,
  logLevel: "silent",
}).outputFiles[0].text;
const moduleShim = { exports: {} };
new Function("module", "exports", "window", "Headers", "FormData", "DOMException", bundle)(
  moduleShim, moduleShim.exports, globalThis, Headers, FormData, DOMException,
);
const { ApiClient, ApiError } = moduleShim.exports;

function abortableFetch(_path, options) {
  return new Promise((_resolve, reject) => {
    options.signal.addEventListener("abort", () => reject(new DOMException("Cancelled", "AbortError")), { once: true });
  });
}

test("caller cancellation and request timeout remain independent", async () => {
  const external = new AbortController();
  const client = new ApiClient({ fetchImpl: abortableFetch, defaultTimeoutMs: 20 });
  await assert.rejects(client.request("/api/slow", { signal: external.signal }), /timed out/);
  assert.equal(external.signal.aborted, false);

  const cancelled = new AbortController();
  const request = client.request("/api/slow", { signal: cancelled.signal, timeoutMs: 1000 });
  cancelled.abort();
  await assert.rejects(request, error => error.name === "AbortError");
});

test("transient GET responses retry but mutations do not", async () => {
  let calls = 0;
  const client = new ApiClient({ fetchImpl: async () => {
    calls += 1;
    return calls === 1 ? new Response("busy", { status: 503 }) : new Response('{"ok":true}');
  } });
  assert.deepEqual(await client.request("/api/status"), { ok: true });
  assert.equal(calls, 2);

  calls = 0;
  const postClient = new ApiClient({ fetchImpl: async () => {
    calls += 1;
    return new Response("busy", { status: 503 });
  } });
  await assert.rejects(postClient.request("/api/action", { method: "POST" }), ApiError);
  assert.equal(calls, 1);
});

test("API client rejects non-API paths", async () => {
  const client = new ApiClient({ fetchImpl: async () => new Response("{}") });
  await assert.rejects(client.request("https://example.com/"), /same-origin/);
});
