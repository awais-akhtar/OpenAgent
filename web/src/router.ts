import { moveFocus, query, queryAll } from "./dom.js";
import type { UiCapability } from "./capabilities.js";

export interface NavigationOptions {
  fromHistory?: boolean;
  replace?: boolean;
  focus?: boolean;
}

export interface ViewRouterOptions {
  capabilities: ReadonlyMap<string, UiCapability>;
  onChange?: (view: string, previousView: string) => void;
}

export class ViewRouter {
  private activeView = "";
  private readonly onChange: ((view: string, previousView: string) => void) | undefined;

  constructor(private readonly options: ViewRouterOptions) {
    this.onChange = options.onChange;
  }

  current(): string {
    return this.activeView;
  }

  navigate(view: string, options: NavigationOptions = {}): boolean {
    const capability = this.options.capabilities.get(view);
    const panel = query<HTMLElement>(`[data-view-panel="${CSS.escape(view)}"]`);
    if (!capability?.available || !panel) return false;

    const previousView = this.activeView;
    this.activeView = view;
    queryAll<HTMLElement>("[data-view-panel]").forEach(candidate => {
      const active = candidate.dataset.viewPanel === view;
      candidate.classList.toggle("active", active);
      candidate.setAttribute("aria-hidden", String(!active));
    });
    queryAll<HTMLElement>("[data-view-link]").forEach(link => {
      const active = link.dataset.viewLink === view;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    this.revealActiveNavigation(view);

    if (!options.fromHistory) {
      const url = new URL(window.location.href);
      url.searchParams.set("view", view);
      window.history[options.replace ? "replaceState" : "pushState"]({ view }, "", url);
    }
    if (options.focus) moveFocus(panel);
    this.onChange?.(view, previousView);
    return true;
  }

  readLocation(fallback = "dashboard"): string {
    const requested = new URL(window.location.href).searchParams.get("view") ?? fallback;
    return this.options.capabilities.get(requested)?.available ? requested : fallback;
  }

  private revealActiveNavigation(view: string): void {
    const activeLink = query<HTMLElement>(`[data-view-link="${CSS.escape(view)}"]`);
    if (!activeLink || !activeLink.classList.contains("nav-group-hidden")) return;
    let group = activeLink.previousElementSibling;
    while (group && !group.classList.contains("nav-group-toggle")) group = group.previousElementSibling;
    if (!(group instanceof HTMLElement)) return;
    group.classList.remove("collapsed");
    group.setAttribute("aria-expanded", "true");
    let sibling = group.nextElementSibling;
    while (sibling && !sibling.classList.contains("nav-group-toggle")) {
      if (sibling.classList.contains("nav-link")) sibling.classList.remove("nav-group-hidden");
      sibling = sibling.nextElementSibling;
    }
  }
}
