export type QueryRoot = Document | DocumentFragment | Element;

export function query<T extends Element = HTMLElement>(selector: string, root: QueryRoot = document): T | null {
  return root.querySelector<T>(selector);
}

export function queryAll<T extends Element = HTMLElement>(selector: string, root: QueryRoot = document): T[] {
  return Array.from(root.querySelectorAll<T>(selector));
}

export function escapeHtml(value: unknown): string {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

export function setText(target: string | Element, value: unknown): void {
  const element = typeof target === "string" ? query(target) : target;
  if (element) element.textContent = String(value ?? "");
}

export function setPressed(element: Element | null, pressed: boolean): void {
  if (!element) return;
  element.classList.toggle("active", pressed);
  element.setAttribute("aria-pressed", String(pressed));
}

export function moveFocus(target: Element | null): void {
  if (!(target instanceof HTMLElement)) return;
  if (!target.hasAttribute("tabindex")) target.tabIndex = -1;
  target.focus({ preventScroll: true });
}

export function closestFromEvent<T extends Element = HTMLElement>(event: Event, selector: string): T | null {
  return event.target instanceof Element ? event.target.closest<T>(selector) : null;
}
