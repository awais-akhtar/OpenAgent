import { escapeHtml, query } from "./dom.js";

export type NotificationTone = "info" | "success" | "warning" | "error";

export class NotificationCenter {
  constructor(private readonly selector = "#toastStack") {}

  show(title: string, body = "", tone: NotificationTone = "info", durationMs = 4200): void {
    const host = query<HTMLElement>(this.selector);
    if (!host) return;
    const node = document.createElement("div");
    node.className = `toast toast-${tone}`;
    node.setAttribute("role", tone === "error" ? "alert" : "status");
    node.innerHTML = `<strong>${escapeHtml(title)}</strong>${body ? `<p>${escapeHtml(body)}</p>` : ""}`;
    host.appendChild(node);
    window.setTimeout(() => node.remove(), durationMs);
  }
}
