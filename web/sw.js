/**
 * OpenAgent Service Worker
 * Caches versioned static assets only. Navigation is never served from cache,
 * so a stopped historical port cannot look like a running OpenAgent instance.
 */

const CACHE_VERSION = "openagent-v25-codeq-chat-20260812";
const SHELL_ASSETS = [
  "/styles.css?v=20260812-codeq-chat-5",
  "/app.js?v=20260812-codeq-chat-5",
  "/manifest.json?v=20260812-codeq-chat-5",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);

  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request, { cache: "no-store" }).catch(() => new Response(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>OpenAgent is not running</title></head><body style=\"font-family:system-ui;padding:40px;max-width:720px;margin:auto\"><h1>OpenAgent is not running at this address</h1><p>Launch OpenAgent Desktop and use <strong>http://127.0.0.1:8765/</strong>.</p></body></html>",
        { status: 503, headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" } },
      ))
    );
    return;
  }

  // Always network-first for API calls
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(event.request).catch(() => new Response(
        JSON.stringify({
          error: "The OpenAgent runtime is not reachable. Close this window and launch OpenAgent again.",
          runtime_unreachable: true,
        }),
        { status: 503, headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" } },
      ))
    );
    return;
  }

  // Network-first for versioned static assets. Only these files may fall back
  // to cache; an HTML navigation never does.
  event.respondWith(
    fetch(event.request).then((response) => {
      if (response && response.status === 200 && response.type === "basic") {
        const clone = response.clone();
        caches.open(CACHE_VERSION).then((cache) => cache.put(event.request, clone));
      }
      return response;
    }).catch(() => caches.match(event.request))
  );
});
