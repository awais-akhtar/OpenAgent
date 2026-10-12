export interface PaletteAction {
  label: string;
  run: () => void;
}

interface PaletteEntry extends PaletteAction {
  group: string;
  icon: string;
}

export class CommandPalette {
  private entries: PaletteEntry[] = [];
  private visible: PaletteEntry[] = [];
  private selected = 0;
  private returnFocus: HTMLElement | null = null;

  constructor(
    private readonly overlay: HTMLElement,
    private readonly input: HTMLInputElement,
    private readonly results: HTMLElement,
    private readonly navigate: (view: string) => void,
    private readonly actions: PaletteAction[] = [],
  ) {
    input.addEventListener("input", () => this.render());
    input.addEventListener("keydown", event => this.onKeyDown(event));
    overlay.addEventListener("click", event => {
      if (event.target === overlay || (event.target as Element).closest("[data-palette-close]")) this.close();
    });
    overlay.addEventListener("keydown", event => {
      if (event.key !== "Tab") return;
      const closeButton = overlay.querySelector<HTMLElement>("[data-palette-close]");
      if (!closeButton) return;
      event.preventDefault();
      (document.activeElement === this.input ? closeButton : this.input).focus();
    });
  }

  get isOpen(): boolean {
    return this.overlay.classList.contains("open");
  }

  open(): void {
    this.returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const views = Array.from(document.querySelectorAll<HTMLAnchorElement>(".app-nav [data-view-link], .sidebar-bottom-nav [data-view-link]"));
    this.entries = views.map(link => ({
      label: link.textContent?.trim() || link.dataset.viewLink || "",
      icon: link.querySelector("use")?.getAttribute("href") || "",
      group: "Navigate",
      run: () => this.navigate(link.dataset.viewLink || "dashboard"),
    }));
    this.entries.push(...this.actions.map(action => ({ ...action, group: "Actions", icon: "" })));
    this.input.value = "";
    this.overlay.classList.add("open");
    this.input.setAttribute("aria-expanded", "true");
    this.selected = 0;
    this.render();
    this.input.focus();
  }

  close(): void {
    if (!this.isOpen) return;
    this.overlay.classList.remove("open");
    this.input.setAttribute("aria-expanded", "false");
    this.returnFocus?.focus();
    this.returnFocus = null;
  }

  private render(): void {
    const query = this.input.value.trim().toLocaleLowerCase();
    this.visible = this.entries.filter(entry => !query || entry.label.toLocaleLowerCase().includes(query));
    this.selected = Math.min(this.selected, Math.max(0, this.visible.length - 1));
    this.results.replaceChildren();
    if (!this.visible.length) {
      const empty = document.createElement("p");
      empty.className = "palette-empty";
      empty.textContent = "No matching commands";
      this.results.append(empty);
      this.input.removeAttribute("aria-activedescendant");
      return;
    }
    this.visible.forEach((entry, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "palette-item";
      button.id = `palette-option-${index}`;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", String(index === this.selected));
      if (index === this.selected) button.classList.add("focused");
      const icon = document.createElement("span");
      icon.className = "pi-icon";
      icon.setAttribute("aria-hidden", "true");
      if (entry.icon) {
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
        use.setAttribute("href", entry.icon);
        svg.append(use);
        icon.append(svg);
      }
      const label = document.createElement("span");
      label.className = "pi-label";
      label.textContent = entry.label;
      const group = document.createElement("span");
      group.className = "pi-group";
      group.textContent = entry.group;
      button.append(icon, label, group);
      button.addEventListener("click", () => this.run(index));
      this.results.append(button);
    });
    this.input.setAttribute("aria-activedescendant", `palette-option-${this.selected}`);
    this.results.querySelector<HTMLElement>(".focused")?.scrollIntoView({ block: "nearest" });
  }

  private run(index: number): void {
    const entry = this.visible[index];
    if (!entry) return;
    this.close();
    entry.run();
  }

  private onKeyDown(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      event.preventDefault();
      this.close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      this.selected = Math.max(0, Math.min(this.visible.length - 1, this.selected + (event.key === "ArrowDown" ? 1 : -1)));
      this.render();
    } else if (event.key === "Enter") {
      event.preventDefault();
      this.run(this.selected);
    }
  }
}
