const MAX_ITEMS = 40;
const MAX_DEPTH = 4;

function label(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, character => character.toUpperCase());
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPresent(value: unknown): boolean {
  return value !== null && value !== undefined && value !== "";
}

function valueNode(value: unknown, depth: number): HTMLElement {
  if (depth >= MAX_DEPTH) {
    const text = document.createElement("span");
    text.textContent = Array.isArray(value) ? `${value.length} items` : isRecord(value) ? `${Object.keys(value).length} fields` : String(value);
    return text;
  }
  if (Array.isArray(value)) {
    const list = document.createElement("ol");
    list.className = "result-list";
    value.slice(0, MAX_ITEMS).forEach(item => {
      const row = document.createElement("li");
      row.append(valueNode(item, depth + 1));
      list.append(row);
    });
    if (value.length > MAX_ITEMS) {
      const more = document.createElement("li");
      more.textContent = `${value.length - MAX_ITEMS} more in raw data`;
      list.append(more);
    }
    return list;
  }
  if (isRecord(value)) {
    const fields = document.createElement("dl");
    fields.className = "result-fields";
    Object.entries(value).filter(([, entry]) => isPresent(entry)).slice(0, MAX_ITEMS).forEach(([key, entry]) => {
      const name = document.createElement("dt");
      name.textContent = label(key);
      const detail = document.createElement("dd");
      detail.append(valueNode(entry, depth + 1));
      fields.append(name, detail);
    });
    return fields;
  }
  const text = document.createElement("span");
  text.textContent = typeof value === "boolean" ? (value ? "Yes" : "No") : String(value ?? "Not available");
  return text;
}

export function renderResult(container: HTMLElement, value: unknown): void {
  container.replaceChildren();
  let content = value;
  if (isRecord(value) && "success" in value) {
    if (value.success === false) {
      container.setAttribute("role", "alert");
      container.classList.add("result-error");
      content = value.error ?? { message: "The action could not be completed." };
    } else {
      container.removeAttribute("role");
      container.classList.remove("result-error");
      content = value.data ?? value;
    }
  } else {
    container.removeAttribute("role");
    container.classList.remove("result-error");
  }
  container.append(valueNode(content, 0));
  if (typeof value === "object" && value !== null) {
    const details = document.createElement("details");
    details.className = "result-raw";
    const summary = document.createElement("summary");
    summary.textContent = "Raw data";
    const raw = document.createElement("pre");
    raw.textContent = JSON.stringify(value, null, 2);
    details.append(summary, raw);
    container.append(details);
  }
}
