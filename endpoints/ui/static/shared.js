// Shared by the chat client (app.js) and the status page (status.js): browser
// storage, the connection settings, API calls and DOM helpers.

export const $ = (id) => document.getElementById(id);
export const STORE = "tabby-ui.";

// ---------------------------------------------------------------------------
// Storage

let storageWarned = false;
let storageErrorHandler = null;

// Called once, the first time something can't be saved (storage full or blocked)
export function onStorageError(handler) {
  storageErrorHandler = handler;
}

export function load(key, fallback) {
  try {
    const raw = localStorage.getItem(STORE + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function save(key, value) {
  try {
    localStorage.setItem(STORE + key, JSON.stringify(value));
  } catch (e) {
    if (!storageWarned) {
      storageWarned = true;
      storageErrorHandler?.(e);
    }
  }
}

export function remove(key) {
  try {
    localStorage.removeItem(STORE + key);
  } catch {
    // Nothing to clean up if storage is unavailable
  }
}

// Connection and display settings, shared by both pages
export const settings = Object.assign({ baseUrl: "", apiKey: "", theme: "", markdown: true }, load("settings", {}));
// Theme names from before the theme menu ("light", "dark") map to their new names
settings.theme = window.tabbyTheme.normalize(settings.theme);

// The server prints links to the pages with the admin key in the URL
// fragment, which the browser never sends to the server. Take the key from
// it, then clear it from the address bar.
export function takeKeyFromLink() {
  const key = new URLSearchParams(location.hash.slice(1)).get("key");
  if (!key) return;

  // The link is for the server that printed it, not one set up earlier
  settings.baseUrl = "";
  settings.apiKey = key;
  save("settings", settings);
  history.replaceState(null, "", location.pathname + location.search);
}

// The status page loads and unloads models; it tells open chat tabs through
// this storage key, whose change event reaches every other tab
export const MODEL_CHANGED_KEY = "model-changed";

export function announceModelChange() {
  save(MODEL_CHANGED_KEY, Date.now());
}

// ---------------------------------------------------------------------------
// API

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export function apiUrl(path) {
  return settings.baseUrl.replace(/\/+$/, "") + path;
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    const detail = body.detail ?? body.error?.message ?? body.error ?? body;
    if (Array.isArray(detail)) {
      return detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("\n");
    }
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

export async function api(path, { method = "GET", body, signal } = {}) {
  const headers = {};
  if (settings.apiKey) headers.Authorization = `Bearer ${settings.apiKey}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(apiUrl(path), {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });

  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  return res;
}

// Yields the data field of each server-sent event in a streaming response
export async function* sseData(res) {
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;

    let match;
    while ((match = /\r\n\r\n|\n\n|\r\r/.exec(buffer))) {
      const block = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);

      const data = block
        .split(/\r\n|\r|\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""));
      if (data.length) yield data.join("\n");
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key in node && typeof value !== "string") node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const ICONS = {
  chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  edit: '<path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  reset: '<path d="M18 6 6 18M6 6l12 12"/>',
  star: '<path d="M12 3l2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3l-5.6 2.9 1.1-6.2L3 9.6l6.2-.9z"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="1"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
};

export function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name];
  return svg;
}

export function newId() {
  // crypto.randomUUID is missing outside secure contexts (plain http on a LAN)
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export function fmt(value, digits = 1) {
  return typeof value === "number" ? value.toFixed(digits) : value;
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // The async clipboard API needs a secure context; fall back for plain http
    const area = el("textarea", { style: "position:fixed;opacity:0" });
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}

export function applyTheme() {
  window.tabbyTheme.apply(settings.theme);
}
