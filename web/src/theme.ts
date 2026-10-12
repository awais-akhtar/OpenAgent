export type ThemePreference = "dark" | "light" | "system";

export interface ThemeControllerOptions {
  storageKey?: string;
  onChange?: (preference: ThemePreference, resolved: "dark" | "light") => void;
}

const preferences = new Set<ThemePreference>(["dark", "light", "system"]);

export class ThemeController {
  private readonly storageKey: string;
  private readonly onChange: ThemeControllerOptions["onChange"];
  private media: MediaQueryList | null = null;
  private mediaListener: (() => void) | null = null;
  preference: ThemePreference = "system";

  constructor(options: ThemeControllerOptions = {}) {
    this.storageKey = options.storageKey ?? "oa-theme";
    this.onChange = options.onChange;
  }

  initialize(): void {
    let saved: string | null = null;
    try { saved = localStorage.getItem(this.storageKey); } catch { /* Storage can be blocked. */ }
    this.apply(saved ?? "system", false);
    if (!window.matchMedia) return;
    this.media = window.matchMedia("(prefers-color-scheme: dark)");
    this.mediaListener = () => {
      if (this.preference === "system") this.apply("system", false);
    };
    if (typeof this.media.addEventListener === "function") this.media.addEventListener("change", this.mediaListener);
    else this.media.addListener(this.mediaListener);
  }

  apply(value: string, persist = true): void {
    const preference = preferences.has(value as ThemePreference) ? value as ThemePreference : "system";
    const resolved = preference === "system"
      ? (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light")
      : preference;
    this.preference = preference;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.themePreference = preference;
    if (persist) {
      try { localStorage.setItem(this.storageKey, preference); } catch { /* Storage can be blocked. */ }
    }
    document.querySelectorAll<HTMLElement>("[data-theme-select]").forEach(button => {
      const selected = button.dataset.themeSelect === preference;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    const select = document.querySelector<HTMLSelectElement>("#settingsTheme");
    if (select) select.value = preference;
    document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
      ?.setAttribute("content", resolved === "dark" ? "#0b0f16" : "#f5f7fb");
    this.onChange?.(preference, resolved);
  }

  destroy(): void {
    if (!this.media || !this.mediaListener) return;
    if (typeof this.media.removeEventListener === "function") this.media.removeEventListener("change", this.mediaListener);
    else this.media.removeListener(this.mediaListener);
  }
}
