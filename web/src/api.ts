export interface ApiErrorBody {
  message?: string;
  code?: string;
  details?: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string;
  readonly details: unknown;

  constructor(message: string, response: Response, body: ApiErrorBody = {}) {
    super(message);
    this.name = "ApiError";
    this.status = response.status;
    this.code = body.code ?? "request_failed";
    this.requestId = response.headers.get("x-request-id") ?? "";
    this.details = body.details;
  }
}

export interface ApiClientOptions {
  getToken?: () => string;
  fetchImpl?: typeof fetch;
  defaultTimeoutMs?: number;
}

export type ApiRequestOptions = RequestInit & { timeoutMs?: number };

export class ApiClient {
  private readonly getToken: () => string;
  private readonly fetchImpl: typeof fetch;
  private readonly defaultTimeoutMs: number;

  constructor(options: ApiClientOptions = {}) {
    this.getToken = options.getToken ?? (() => "");
    this.fetchImpl = options.fetchImpl ?? window.fetch.bind(window);
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? 45_000;
  }

  async request<T = unknown>(path: string, options: ApiRequestOptions = {}): Promise<T> {
    if (!path.startsWith("/api/")) throw new Error("API requests must use a same-origin /api/ path.");
    const controller = new AbortController();
    const { timeoutMs = this.defaultTimeoutMs, signal, ...requestOptions } = options;
    const onAbort = () => controller.abort(signal?.reason);
    if (signal?.aborted) onAbort();
    else signal?.addEventListener("abort", onAbort, { once: true });
    const timeout = window.setTimeout(() => controller.abort("request_timeout"), timeoutMs);
    const token = this.getToken();
    const headers = new Headers(requestOptions.headers);
    if (requestOptions.body && !(requestOptions.body instanceof FormData) && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const method = (requestOptions.method ?? "GET").toUpperCase();
    const retries = method === "GET" || method === "HEAD" ? 1 : 0;

    try {
      for (let attempt = 0; attempt <= retries; attempt += 1) {
        if (controller.signal.aborted) throw new DOMException("Request cancelled", "AbortError");
        try {
          const response = await this.fetchImpl(path, { ...requestOptions, headers, signal: controller.signal });
          if (attempt < retries && [429, 502, 503, 504].includes(response.status)) {
            await response.body?.cancel();
            continue;
          }
          const raw = await response.text();
          let body: unknown = {};
          if (raw) {
            try { body = JSON.parse(raw); }
            catch { body = { message: raw }; }
          }
          if (!response.ok) {
            const record = typeof body === "object" && body !== null ? body as Record<string, unknown> : {};
            const nested = typeof record.error === "object" && record.error !== null ? record.error as ApiErrorBody : {};
            const direct = typeof record.error === "string" ? record.error : undefined;
            throw new ApiError(direct ?? nested.message ?? `Request failed: ${response.status}`, response, nested);
          }
          return body as T;
        } catch (error) {
          if (attempt < retries && error instanceof TypeError && !controller.signal.aborted) continue;
          throw error;
        }
      }
      throw new Error("Request attempts exhausted.");
    } catch (error) {
      if (controller.signal.aborted && controller.signal.reason === "request_timeout") {
        throw new Error("The request timed out.");
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
    }
  }
}
