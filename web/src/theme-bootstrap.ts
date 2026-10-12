const allowedThemes = new Set(["dark", "light", "system"]);

try {
  const stored = localStorage.getItem("oa-theme") ?? "system";
  const preference = allowedThemes.has(stored) ? stored : "system";
  const systemDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  const resolved = preference === "system" ? (systemDark ? "dark" : "light") : preference;
  document.documentElement.dataset.theme = resolved;
  document.documentElement.dataset.themePreference = preference;
} catch {
  document.documentElement.dataset.theme = "dark";
  document.documentElement.dataset.themePreference = "system";
}
