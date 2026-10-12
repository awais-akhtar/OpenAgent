export interface UiCapability {
  id: string;
  label: string;
  path: string;
  available: boolean;
}

export function discoverCapabilities(root: ParentNode = document): UiCapability[] {
  return Array.from(root.querySelectorAll<HTMLElement>("[data-view-panel]")).map(panel => {
    const id = panel.dataset.viewPanel ?? "";
    const link = root.querySelector<HTMLAnchorElement>(`[data-view-link="${CSS.escape(id)}"]`);
    return {
      id,
      label: link?.textContent?.trim() || panel.getAttribute("aria-label") || id,
      path: link?.getAttribute("href") || `/?view=${encodeURIComponent(id)}`,
      available: !panel.hasAttribute("data-capability-disabled"),
    };
  });
}

export function capabilityMap(capabilities: readonly UiCapability[]): ReadonlyMap<string, UiCapability> {
  return new Map(capabilities.map(capability => [capability.id, capability]));
}
