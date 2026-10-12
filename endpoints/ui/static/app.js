// TabbyAPI chat client. Plain browser JavaScript with no build step: it calls
// the regular API (chat completions, model and sampler override endpoints) and
// keeps its chats and settings in the browser's local storage.

const $ = (id) => document.getElementById(id);
const STORE = "tabby-ui.";

// ---------------------------------------------------------------------------
// Storage

let storageWarned = false;

function load(key, fallback) {
  try {
    const raw = localStorage.getItem(STORE + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(STORE + key, JSON.stringify(value));
  } catch (e) {
    if (!storageWarned) {
      storageWarned = true;
      setStatus(`Could not save to browser storage (${e.name}). Delete some chats to free space.`, true);
    }
  }
}

function remove(key) {
  try {
    localStorage.removeItem(STORE + key);
  } catch {
    // Nothing to clean up if storage is unavailable
  }
}

// ---------------------------------------------------------------------------
// State

const settings = Object.assign({ baseUrl: "", apiKey: "", theme: "", markdown: true }, load("settings", {}));
let params = load("params", {});
// Model names loaded through the client, newest first. The model list only has
// the top-level folders of the model directory, so nested ones (typed as a
// path) would otherwise have to be typed again every time.
let recentModels = load("recent-models", []);
// Starred model names, shown at the top of the model list
let favoriteModels = load("favorite-models", []);
let sessionIndex = load("sessions", []);
let session = null;

const state = {
  permission: null,
  models: [],
  currentModel: null,
  // Full path of the loaded model, and its name in the model list if listed
  modelPath: null,
  loadedName: null,
  selectedModel: null,
  loading: false,
  overrides: null,
  generating: null,
};

let samplerGroups = [];
const paramSpecs = new Map();

// ---------------------------------------------------------------------------
// API

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function apiUrl(path) {
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

async function api(path, { method = "GET", body, signal } = {}) {
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
async function* sseData(res) {
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

function el(tag, attrs = {}, ...children) {
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
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name];
  return svg;
}

function newId() {
  // crypto.randomUUID is missing outside secure contexts (plain http on a LAN)
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function fmt(value, digits = 1) {
  return typeof value === "number" ? value.toFixed(digits) : value;
}

function setStatus(text, isError = false) {
  const line = $("status-line");
  line.textContent = text || "";
  line.classList.toggle("error", isError);
}

async function copyText(text) {
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

// ---------------------------------------------------------------------------
// Markdown

// markdown-it (vendor/markdown-it.min.js) with its safe defaults: HTML in a
// reply is shown as text, and javascript:, vbscript: and file: links are
// refused. Single newlines are kept as line breaks, as models tend to mean them.
const markdown = window.markdownit ? window.markdownit({ linkify: true, breaks: true }) : null;

if (markdown) {
  // Links open in a new tab, without a handle back to this page
  const renderLinkOpen =
    markdown.renderer.rules.link_open ||
    ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  markdown.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    tokens[idx].attrSet("target", "_blank");
    tokens[idx].attrSet("rel", "noopener noreferrer");
    return renderLinkOpen(tokens, idx, options, env, self);
  };

  // The page's security policy only loads images embedded as data: URLs, so
  // other images become links, and a reply can't make the browser fetch them
  const renderImage = markdown.renderer.rules.image;
  markdown.renderer.rules.image = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const src = token.attrGet("src") || "";
    if (src.startsWith("data:image/")) return renderImage(tokens, idx, options, env, self);

    const alt = self.renderInlineAsText(token.children || [], options, env) || src;
    const href = markdown.utils.escapeHtml(src);
    return `<a class="remote-image" href="${href}" target="_blank" rel="noopener noreferrer">${markdown.utils.escapeHtml(alt)}</a>`;
  };
}

// Code blocks get a header with the language and a copy button. SVG code
// also gets a preview frame below it, filled in by addSvgPreviews.
const SVG_SOURCE = /^\s*(<\?xml[^>]*>\s*)?(<!--[\s\S]*?-->\s*)*<svg[\s>]/i;

if (markdown) {
  const renderFence = markdown.renderer.rules.fence;
  markdown.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const lang = token.info.trim().split(/\s+/)[0];
    const isSvg = lang.toLowerCase() === "svg" || SVG_SOURCE.test(token.content);
    const escape = markdown.utils.escapeHtml;

    return (
      `<div class="code-block${isSvg ? " svg" : ""}">` +
      `<div class="code-head"><span class="code-lang">${escape(lang)}</span>` +
      `<button type="button" class="code-copy">Copy</button></div>` +
      renderFence(tokens, idx, options, env, self) +
      (isSvg ? '<div class="svg-preview"></div>' : "") +
      "</div>"
    );
  };
}

// Preview images by SVG source. Replies re-render on every streamed token, so
// reusing the same image element keeps a finished preview from flickering.
const svgPreviews = new Map();

function svgPreview(source) {
  let image = svgPreviews.get(source);
  if (image) {
    svgPreviews.delete(source);
  } else {
    // As an image, an SVG can't run scripts or load anything, whatever it holds
    let svg = source.trim();
    if (!/<svg[^>]*\sxmlns\s*=/i.test(svg)) svg = svg.replace(/<svg/i, '<svg xmlns="http://www.w3.org/2000/svg"');
    if (/\bxlink:/.test(svg) && !/xmlns:xlink\s*=/.test(svg)) {
      svg = svg.replace(/<svg/i, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
    }
    image = el("img", { alt: "SVG preview" });
    image.addEventListener("error", () => {
      image.dataset.failed = "true";
      image.replaceWith(svgError());
    });
    image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }
  // Most recently used last, so the oldest go first
  svgPreviews.set(source, image);
  if (svgPreviews.size > 64) svgPreviews.delete(svgPreviews.keys().next().value);
  return image;
}

function svgError() {
  return el("p", { class: "svg-error" }, "This SVG could not be displayed.");
}

function addSvgPreviews(node) {
  for (const frame of node.querySelectorAll(".svg-preview")) {
    const source = frame.parentElement.querySelector("code")?.textContent || "";
    // A block still streaming in has no closing tag yet
    if (/<\/svg>\s*$/i.test(source)) {
      const image = svgPreview(source);
      frame.append(image.dataset.failed ? svgError() : image);
    } else frame.append(el("p", { class: "svg-pending" }, "Preview appears when the SVG is complete."));
  }
}

// Only replies are formatted; the raw setting shows exactly what the model wrote
function setContent(node, msg) {
  if (markdown && settings.markdown && msg.role === "assistant") {
    node.classList.add("markdown");
    node.innerHTML = markdown.render(msg.content);
    addSvgPreviews(node);
  } else {
    node.classList.remove("markdown");
    node.textContent = msg.content;
  }
}

// ---------------------------------------------------------------------------
// Images

// Formats the server takes as data: URLs (its check only allows plain
// subtypes, so no image/svg+xml or image/x-ms-bmp)
const IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"]);
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
// Well below the ~179 megapixels at which PIL refuses an image outright
const MAX_IMAGE_PIXELS = 50_000_000;
const MAX_IMAGES_PER_MESSAGE = 16;

// Images waiting to be sent with the next message
let attachments = [];

// Image data lives in IndexedDB, since local storage (where the chats are)
// holds only a few megabytes. Messages refer to images by id. Without
// IndexedDB (some private windows) images last until the page closes.
const imageBlobs = new Map();
const imageUrls = new Map();
let imageDb = null;

function openImageDb() {
  imageDb ??= new Promise((resolve) => {
    try {
      const request = indexedDB.open("tabby-ui", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("images");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
  return imageDb;
}

async function imageDbRequest(mode, action) {
  const db = await openImageDb();
  if (!db) return undefined;
  return new Promise((resolve, reject) => {
    const tx = db.transaction("images", mode);
    const request = action(tx.objectStore("images"));
    tx.oncomplete = () => resolve(request?.result);
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}

async function storeImage(id, blob) {
  imageBlobs.set(id, blob);
  try {
    await imageDbRequest("readwrite", (store) => store.put(blob, id));
  } catch (e) {
    setStatus(`Could not save an image in browser storage (${e?.name || e}); it lasts until the page closes.`, true);
  }
}

async function getImage(id) {
  if (!imageBlobs.has(id)) {
    try {
      const blob = await imageDbRequest("readonly", (store) => store.get(id));
      if (blob) imageBlobs.set(id, blob);
    } catch {
      // Treated as missing
    }
  }
  return imageBlobs.get(id) || null;
}

async function imageUrl(id) {
  if (!imageUrls.has(id)) {
    const blob = await getImage(id);
    if (!blob) return null;
    imageUrls.set(id, URL.createObjectURL(blob));
  }
  return imageUrls.get(id);
}

function readDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

// Delete stored images that no chat refers to any more
async function collectImages() {
  const used = new Set(attachments.map((a) => a.id));
  for (const item of sessionIndex) {
    const s = item.id === session?.id ? session : load(`session.${item.id}`, null);
    for (const m of s?.messages || []) for (const id of m.images || []) used.add(id);
  }

  for (const id of [...imageBlobs.keys()]) {
    if (used.has(id)) continue;
    imageBlobs.delete(id);
    if (imageUrls.has(id)) URL.revokeObjectURL(imageUrls.get(id));
    imageUrls.delete(id);
  }

  try {
    const keys = (await imageDbRequest("readonly", (store) => store.getAllKeys())) || [];
    const unused = keys.filter((key) => !used.has(key));
    if (unused.length) await imageDbRequest("readwrite", (store) => unused.forEach((key) => store.delete(key)));
  } catch {
    // Try again next time
  }
}

function formatBytes(bytes) {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`;
}

// The image's size from its header, without decoding the pixels
function imageDimensions(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve([image.naturalWidth, image.naturalHeight]);
    image.onerror = reject;
    image.src = url;
  });
}

function warnIfNoVision() {
  if (state.currentModel && !state.currentModel.parameters?.use_vision) {
    setStatus("The loaded model has no vision support, so the server will ignore images.", true);
    return true;
  }
  return false;
}

async function addAttachments(files) {
  const problems = [];
  for (const file of files) {
    const name = file.name || "Pasted image";
    if (attachments.length >= MAX_IMAGES_PER_MESSAGE) {
      problems.push(`At most ${MAX_IMAGES_PER_MESSAGE} images per message`);
      break;
    }
    // Type and size come from the file listing, so nothing is read for these
    if (!IMAGE_TYPES.has(file.type)) {
      problems.push(`${name} is not a PNG, JPEG, WebP, GIF or BMP image`);
      continue;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      problems.push(`${name} is ${formatBytes(file.size)}, over the ${formatBytes(MAX_IMAGE_BYTES)} limit`);
      continue;
    }

    const url = URL.createObjectURL(file);
    let width, height;
    try {
      [width, height] = await imageDimensions(url);
    } catch {
      URL.revokeObjectURL(url);
      problems.push(`${name} could not be read as an image`);
      continue;
    }
    if (width * height > MAX_IMAGE_PIXELS) {
      URL.revokeObjectURL(url);
      problems.push(`${name} is ${width}×${height}, over ${MAX_IMAGE_PIXELS / 1e6} megapixels`);
      continue;
    }

    attachments.push({ id: newId(), blob: file, url, name, width, height });
  }

  renderAttachments();
  if (problems.length) setStatus(problems.join(". ") + ".", true);
  else if (attachments.length && !warnIfNoVision()) setStatus("");
}

function removeAttachment(id) {
  const index = attachments.findIndex((a) => a.id === id);
  if (index < 0) return;
  URL.revokeObjectURL(attachments[index].url);
  attachments.splice(index, 1);
  renderAttachments();
}

function renderAttachments() {
  const strip = $("attachments");
  strip.hidden = !attachments.length;
  strip.replaceChildren(
    ...attachments.map((a) =>
      el(
        "div",
        { class: "attachment", title: `${a.name}, ${a.width}×${a.height}, ${formatBytes(a.blob.size)}` },
        el("img", { src: a.url, alt: a.name }),
        el(
          "button",
          { type: "button", class: "attachment-remove", "aria-label": `Remove ${a.name}`, onclick: () => removeAttachment(a.id) },
          icon("reset"),
        ),
      ),
    ),
  );
}

function openImageViewer(url) {
  const viewer = $("image-viewer");
  viewer.querySelector("img").src = url;
  viewer.showModal();
}

function renderImages(ids) {
  return el(
    "div",
    { class: "message-images" },
    ids.map((id) => {
      const thumb = el("img", { alt: "Attached image", class: "thumb" });
      // Usually cached, but a reload has to fetch it from IndexedDB first
      const cached = imageUrls.get(id);
      if (cached) thumb.src = cached;
      imageUrl(id).then((url) => {
        if (!url) thumb.replaceWith(el("span", { class: "thumb missing" }, "Image no longer stored"));
        else if (!cached) thumb.src = url;
      });
      thumb.addEventListener("click", () => thumb.src && openImageViewer(thumb.src));
      return thumb;
    }),
  );
}

// ---------------------------------------------------------------------------
// Sessions

function sessionSummary(s) {
  return { id: s.id, title: s.title, updated: s.updated };
}

function persistSession(s = session) {
  s.updated = Date.now();
  save(`session.${s.id}`, s);
  sessionIndex = [sessionSummary(s), ...sessionIndex.filter((x) => x.id !== s.id)];
  save("sessions", sessionIndex);
  renderSessionList();
}

function createSession() {
  const s = {
    id: newId(),
    title: "New chat",
    // Carry the system prompt over: test runs usually reuse the same setup
    system: session?.system || "",
    messages: [],
    created: Date.now(),
    updated: Date.now(),
  };
  persistSession(s);
  return s;
}

function openSession(id) {
  // A chat that is still streaming lives in memory, ahead of its stored copy
  const s = state.generating?.session.id === id ? state.generating.session : load(`session.${id}`, null);
  if (!s) {
    sessionIndex = sessionIndex.filter((x) => x.id !== id);
    save("sessions", sessionIndex);
    return false;
  }
  session = s;
  save("current", id);
  renderSessionList();
  renderThread();
  closeDrawers();
  return true;
}

function newChat() {
  if (session && session.messages.length === 0) {
    $("input").focus();
    return;
  }
  session = createSession();
  save("current", session.id);
  renderSessionList();
  renderThread();
  closeDrawers();
  $("input").focus();
}

function deleteSession(id) {
  if (state.generating?.session.id === id) {
    state.generating.session.deleted = true;
    state.generating.controller.abort();
  }
  remove(`session.${id}`);
  sessionIndex = sessionIndex.filter((x) => x.id !== id);
  save("sessions", sessionIndex);
  collectImages();

  if (session?.id === id) {
    session = null;
    if (!sessionIndex.length || !openSession(sessionIndex[0].id)) {
      session = createSession();
      save("current", session.id);
      renderThread();
    }
  }
  renderSessionList();
}

function renameSession(id, title) {
  const s = id === session?.id ? session : load(`session.${id}`, null);
  if (!s) return;
  s.title = title.trim() || s.title;
  persistSession(s);
}

function renderSessionList() {
  const list = $("session-list");
  list.replaceChildren();

  for (const item of sessionIndex) {
    const row = el(
      "div",
      {
        class: "session" + (item.id === session?.id ? " active" : ""),
        title: item.title,
        onclick: () => item.id !== session?.id && openSession(item.id),
      },
      icon("chat"),
    );
    const title = el("span", { class: "session-title" }, item.title);

    const startRename = (e) => {
      e.stopPropagation();
      const input = el("input", { type: "text", value: item.title });
      const finish = (commit) => {
        if (commit) renameSession(item.id, input.value);
        else renderSessionList();
      };
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") finish(true);
        if (ev.key === "Escape") finish(false);
      });
      input.addEventListener("blur", () => finish(true));
      input.addEventListener("click", (ev) => ev.stopPropagation());
      title.replaceChildren(input);
      input.focus();
      input.select();
    };

    const actions = el(
      "span",
      { class: "session-actions" },
      el("button", { type: "button", title: "Rename", onclick: startRename }, icon("edit")),
      el(
        "button",
        {
          type: "button",
          title: "Delete",
          onclick: (e) => {
            e.stopPropagation();
            if (confirm(`Delete "${item.title}"?`)) deleteSession(item.id);
          },
        },
        icon("trash"),
      ),
    );

    title.addEventListener("dblclick", startRename);
    row.append(title, actions);
    list.append(row);
  }
}

// ---------------------------------------------------------------------------
// Thread

// DOM nodes of the message being streamed, so tokens update it in place
let live = null;
let renderQueued = false;

function nearBottom() {
  const t = $("thread");
  return t.scrollHeight - t.scrollTop - t.clientHeight < 80;
}

function scrollToBottom() {
  const t = $("thread");
  t.scrollTop = t.scrollHeight;
}

function statsText(msg) {
  const parts = [];
  const u = msg.usage;
  if (u) {
    let gen = `${u.completion_tokens} tok`;
    if (u.completion_tokens_per_sec != null) gen += ` @ ${fmt(u.completion_tokens_per_sec)} tok/s`;
    parts.push(gen);

    let prompt = `prompt ${u.prompt_tokens}`;
    const cached = u.prompt_tokens_details?.cached_tokens;
    if (cached) prompt += ` (${cached} cached)`;
    if (u.prompt_tokens_per_sec != null) prompt += ` @ ${fmt(u.prompt_tokens_per_sec)} tok/s`;
    parts.push(prompt);
  }
  if (msg.finish) parts.push(msg.finish);
  if (msg.model) parts.push(msg.model);
  return parts.join(" · ");
}

function renderMessage(msg, index) {
  const streaming = state.generating?.message === msg;
  const body = el("div", { class: "message-body" + (streaming ? " streaming" : "") });

  let reasoningText = null;
  let reasoningBox = null;
  if (msg.role === "assistant" && (msg.reasoning || streaming)) {
    reasoningText = el("div", { class: "reasoning-text" }, msg.reasoning || "");
    reasoningBox = el(
      "details",
      { class: "reasoning", open: !msg.content },
      el("summary", {}, "Reasoning"),
      reasoningText,
    );
    reasoningBox.hidden = !msg.reasoning;
    body.append(reasoningBox);
  }

  if (msg.images?.length) body.append(renderImages(msg.images));

  const content = el("div", { class: "content" });
  setContent(content, msg);
  body.append(content);

  const error = el("div", { class: "error-text" }, msg.error || "");
  error.hidden = !msg.error;
  body.append(error);

  const stats = el("span", { class: "stats" }, statsText(msg));
  const actions = el("span", { class: "message-actions" });

  if (!streaming) {
    const copyButton = el("button", { type: "button", title: "Copy" }, icon("copy"));
    copyButton.addEventListener("click", async () => {
      await copyText(msg.content);
      copyButton.replaceChildren(icon("check"));
      setTimeout(() => copyButton.replaceChildren(icon("copy")), 1200);
    });
    actions.append(
      copyButton,
      el("button", { type: "button", title: "Edit", onclick: () => editMessage(index, body) }, icon("edit")),
      el("button", { type: "button", title: "Delete", onclick: () => deleteMessage(index) }, icon("trash")),
    );
  }

  body.append(el("div", { class: "message-meta" }, stats, actions));

  if (streaming) live = { body, content, reasoningText, reasoningBox, error, stats };

  return el(
    "div",
    { class: `row row-${msg.role}` },
    el(
      "div",
      { class: "row-inner" },
      el("div", { class: "avatar", "aria-hidden": "true" }, msg.role === "user" ? "U" : "T"),
      body,
    ),
  );
}

function renderThread() {
  live = null;
  const container = $("messages");
  container.replaceChildren(...session.messages.map(renderMessage));

  $("system-prompt").value = session.system || "";
  updateSystemPreview();
  $("empty-state").hidden = session.messages.length > 0;
  updateComposer();
  scrollToBottom();
}

function updateLive() {
  renderQueued = false;
  const gen = state.generating;
  if (!live || !gen || gen.session !== session) return;

  const stick = nearBottom();
  const msg = gen.message;
  setContent(live.content, msg);
  if (live.reasoningText) {
    live.reasoningText.textContent = msg.reasoning;
    live.reasoningBox.hidden = !msg.reasoning;
    if (msg.content && !gen.collapsedReasoning) {
      live.reasoningBox.open = false;
      gen.collapsedReasoning = true;
    }
  }
  live.stats.textContent = statsText(msg);
  if (stick) scrollToBottom();
}

function queueLiveUpdate() {
  if (!renderQueued) {
    renderQueued = true;
    requestAnimationFrame(updateLive);
  }
}

function updateSystemPreview() {
  const text = (session.system || "").replace(/\s+/g, " ").trim();
  $("system-preview").textContent = text ? `— ${text.slice(0, 80)}` : "";
}

function updateComposer() {
  const busy = !!state.generating;
  const last = session.messages[session.messages.length - 1];
  $("stop").hidden = !busy;
  $("regenerate").hidden = busy || !last;
  $("send").disabled = busy;
}

function deleteMessage(index) {
  if (state.generating?.session === session) return;
  session.messages.splice(index, 1);
  persistSession();
  renderThread();
  collectImages();
}

function editMessage(index, body) {
  if (state.generating?.session === session) return;
  const msg = session.messages[index];
  const area = el("textarea", {});
  area.value = msg.content;

  const finish = () => renderThread();
  const commit = (resubmit) => {
    msg.content = area.value;
    if (resubmit) session.messages.length = index + 1;
    persistSession();
    renderThread();
    if (resubmit) {
      collectImages();
      generate();
    }
  };

  const buttons = el(
    "div",
    { class: "button-row" },
    msg.role === "user"
      ? el("button", { type: "button", class: "primary", onclick: () => commit(true) }, "Save & submit")
      : null,
    el("button", { type: "button", class: msg.role === "user" ? "" : "primary", onclick: () => commit(false) }, "Save"),
    el("button", { type: "button", onclick: finish }, "Cancel"),
  );

  area.addEventListener("keydown", (e) => {
    if (e.key === "Escape") finish();
  });

  body.replaceChildren(el("div", { class: "edit-box" }, area, buttons));
  area.style.height = Math.min(Math.max(area.scrollHeight + 4, 80), 480) + "px";
  area.focus();
}

// ---------------------------------------------------------------------------
// Generation

async function buildMessages(s) {
  const messages = [];
  let missing = 0;
  if (s.system && s.system.trim()) messages.push({ role: "system", content: s.system });
  for (const m of s.messages) {
    // Skip failed or stopped-before-output turns
    if (m.role === "assistant" && !m.content) continue;
    if (!m.images?.length) {
      messages.push({ role: m.role, content: m.content });
      continue;
    }

    const parts = [];
    for (const id of m.images) {
      const blob = await getImage(id);
      if (blob) parts.push({ type: "image_url", image_url: { url: await readDataUrl(blob) } });
      else missing++;
    }
    if (m.content) parts.push({ type: "text", text: m.content });
    messages.push({ role: m.role, content: parts });
  }
  if (missing) setStatus(`${missing} image${missing === 1 ? " is" : "s are"} no longer stored and left out.`, true);
  return messages;
}

async function generate() {
  if (state.generating) return;
  const s = session;

  let samplerParams;
  try {
    samplerParams = collectParams();
  } catch (e) {
    setStatus(e.message, true);
    return;
  }

  const msg = { role: "assistant", content: "", reasoning: "" };
  s.messages.push(msg);

  const controller = new AbortController();
  state.generating = { session: s, message: msg, controller, collapsedReasoning: false };
  setStatus("");
  renderThread();

  try {
    const body = {
      messages: await buildMessages({ ...s, messages: s.messages.slice(0, -1) }),
      stream: true,
      stream_options: { include_usage: true },
      ...samplerParams,
    };
    if (s.messages.some((m) => m.images?.length)) warnIfNoVision();

    const res = await api("/v1/chat/completions", { method: "POST", body, signal: controller.signal });

    for await (const data of sseData(res)) {
      if (data === "[DONE]") break;

      let chunk;
      try {
        chunk = JSON.parse(data);
      } catch {
        continue;
      }
      if (chunk.error) throw new Error(chunk.error.message || JSON.stringify(chunk.error));

      if (chunk.model) msg.model = chunk.model;
      const choice = chunk.choices?.[0];
      if (choice) {
        const delta = choice.delta || {};
        if (delta.reasoning_content) msg.reasoning += delta.reasoning_content;
        if (delta.content) msg.content += delta.content;
        if (choice.finish_reason) msg.finish = choice.finish_reason;
      }
      if (chunk.usage) msg.usage = chunk.usage;
      queueLiveUpdate();
    }
  } catch (e) {
    if (controller.signal.aborted) {
      msg.finish = "stopped";
    } else {
      msg.error = e.message || String(e);
      if (e.status === 401) openConnectionDialog();
    }
  } finally {
    if (!msg.reasoning) delete msg.reasoning;
    state.generating = null;
    if (!s.deleted) persistSession(s);
    if (session === s) renderThread();
    if (msg.error) setStatus(msg.error, true);
  }
}

async function send() {
  if (state.generating) return;
  const input = $("input");
  const text = input.value;
  const last = session.messages[session.messages.length - 1];

  if (!text.trim() && !attachments.length) {
    // An empty send answers a trailing user message, e.g. after an edit
    if (last?.role === "user") generate();
    return;
  }

  const sent = attachments;
  attachments = [];
  renderAttachments();
  for (const a of sent) {
    imageUrls.set(a.id, a.url);
    await storeImage(a.id, a.blob);
  }

  const msg = { role: "user", content: text };
  if (sent.length) msg.images = sent.map((a) => a.id);
  session.messages.push(msg);
  if (session.title === "New chat") {
    session.title = text.replace(/\s+/g, " ").trim().slice(0, 48) || (sent.length > 1 ? "Images" : "Image");
  }
  input.value = "";
  autosize();
  persistSession();
  generate();
}

function regenerate() {
  if (state.generating) return;
  const last = session.messages[session.messages.length - 1];
  if (last?.role === "assistant") session.messages.pop();
  if (!session.messages.length) {
    renderThread();
    return;
  }
  generate();
}

function autosize() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

// ---------------------------------------------------------------------------
// Sampling parameters

function unescapeLine(line) {
  return line.replace(/\\(n|t|r|\\)/g, (_, c) => ({ n: "\n", t: "\t", r: "\r", "\\": "\\" })[c]);
}

function paramValue(spec, raw) {
  switch (spec.type) {
    case "int":
    case "float": {
      const n = Number(raw);
      if (raw === "" || Number.isNaN(n)) return undefined;
      return spec.type === "int" ? Math.round(n) : n;
    }
    case "bool":
      return raw === true || raw === "true";
    case "lines": {
      const lines = String(raw).split("\n").filter((l) => l !== "").map(unescapeLine);
      return lines.length ? lines : undefined;
    }
    case "ints": {
      const ids = String(raw).split(/[\s,]+/).filter(Boolean).map(Number);
      if (ids.some((n) => !Number.isInteger(n))) throw new Error(`${spec.label}: token IDs must be integers`);
      return ids.length ? ids : undefined;
    }
    case "json": {
      if (!String(raw).trim()) return undefined;
      try {
        return JSON.parse(raw);
      } catch (e) {
        throw new Error(`${spec.label}: invalid JSON (${e.message})`);
      }
    }
    default:
      return raw === "" ? undefined : raw;
  }
}

function collectParams() {
  const out = {};
  for (const [key, raw] of Object.entries(params)) {
    const spec = paramSpecs.get(key);
    if (!spec) continue;
    const value = paramValue(spec, raw);
    if (value !== undefined) out[key] = value;
  }
  return out;
}

function setParam(key, raw) {
  if (raw === undefined || raw === "") delete params[key];
  else params[key] = raw;
  save("params", params);
}

function serverDefault(spec) {
  const override = state.overrides?.effective?.[spec.key];
  if (override && "override" in override) {
    return { value: override.override, server: true, force: !!override.force };
  }
  return { value: spec.default, server: false, force: false };
}

function defaultLabel(spec) {
  const d = serverDefault(spec);
  let value = d.value;
  if (Array.isArray(value)) value = value.length ? value.join(", ") : "none";
  if (typeof value === "object" && value !== null) value = JSON.stringify(value);
  if (value === "" || value === undefined || value === null) value = "none";
  if (typeof value === "boolean") value = value ? "on" : "off";
  return d.server ? `${value} (server${d.force ? ", forced" : ""})` : String(value);
}

function renderParam(spec) {
  const isSet = () => params[spec.key] !== undefined;
  const wrap = el("div", { class: "param" });
  const id = `param-${spec.key}`;
  const label = el("label", { for: id, title: spec.key }, spec.label);
  if (serverDefault(spec).server) {
    label.append(el("span", { class: "server-tag", title: `Server default: ${defaultLabel(spec)}` }, "server"));
  }
  const reset = el("button", { type: "button", class: "param-reset", title: "Use default" }, icon("reset"));
  const head = el("div", { class: "param-head" }, label);
  const def = defaultLabel(spec);
  let refresh;

  if (spec.type === "int" || spec.type === "float") {
    const number = el("input", {
      id,
      type: "number",
      step: spec.step ?? (spec.type === "int" ? 1 : "any"),
      min: spec.min,
      max: spec.max,
      placeholder: String(serverDefault(spec).value),
      title: `Default: ${def}`,
    });
    head.append(number, reset);
    wrap.append(head);

    let range = null;
    if (spec.min !== undefined && spec.max !== undefined) {
      range = el("input", { type: "range", min: spec.min, max: spec.max, step: spec.step ?? 1, "aria-label": spec.label });
      wrap.append(range);
      range.addEventListener("input", () => {
        number.value = range.value;
        setParam(spec.key, range.value);
        refresh();
      });
    }

    number.addEventListener("input", () => {
      setParam(spec.key, number.value);
      if (range && number.value !== "") range.value = number.value;
      refresh(false);
    });

    refresh = (syncNumber = true) => {
      wrap.classList.toggle("set", isSet());
      if (syncNumber) number.value = isSet() ? params[spec.key] : "";
      if (range) {
        const fallback = Number(serverDefault(spec).value);
        range.value = isSet() ? params[spec.key] : Number.isFinite(fallback) ? fallback : spec.min;
      }
    };
  } else if (spec.type === "bool") {
    const select = el(
      "select",
      { id },
      el("option", { value: "" }, `Default (${def.replace(/ \(server.*\)$/, "")})`),
      el("option", { value: "true" }, "On"),
      el("option", { value: "false" }, "Off"),
    );
    head.append(select, reset);
    wrap.append(head);
    select.addEventListener("change", () => {
      setParam(spec.key, select.value);
      refresh();
    });
    refresh = () => {
      wrap.classList.toggle("set", isSet());
      select.value = isSet() ? String(params[spec.key]) : "";
    };
  } else if (spec.type === "string") {
    const input = el("input", { id, type: "text", class: "wide", placeholder: def === "none" ? "" : def });
    head.append(reset);
    wrap.append(head, input);
    input.addEventListener("input", () => {
      setParam(spec.key, input.value);
      refresh(false);
    });
    refresh = (sync = true) => {
      wrap.classList.toggle("set", isSet());
      if (sync) input.value = isSet() ? params[spec.key] : "";
    };
  } else {
    const area = el("textarea", {
      id,
      rows: spec.type === "text" || spec.type === "json" ? 3 : 2,
      spellcheck: "false",
      placeholder: def === "none" ? "" : `Default: ${def}`,
    });
    head.append(reset);
    wrap.append(head, area);
    const validate = () => {
      let ok = true;
      if (spec.type === "json" || spec.type === "ints") {
        try {
          paramValue(spec, area.value);
        } catch {
          ok = false;
        }
      }
      area.classList.toggle("invalid", !ok);
    };
    area.addEventListener("input", () => {
      setParam(spec.key, area.value);
      validate();
      refresh(false);
    });
    refresh = (sync = true) => {
      wrap.classList.toggle("set", isSet());
      if (sync) area.value = isSet() ? params[spec.key] : "";
      validate();
    };
  }

  if (spec.hint) wrap.append(el("p", { class: "hint" }, spec.hint));

  reset.addEventListener("click", () => {
    setParam(spec.key, undefined);
    refresh();
    updateGroupCounts();
  });
  wrap.addEventListener("input", updateGroupCounts);
  wrap.addEventListener("change", updateGroupCounts);

  refresh();
  return wrap;
}

function updateGroupCounts() {
  for (const group of samplerGroups) {
    const n = group.params.filter((p) => params[p.key] !== undefined).length;
    group.countEl.textContent = n ? `${n} set` : "";
  }
}

function renderParams() {
  const container = $("params");
  const openState = load("groups-open", {});
  container.replaceChildren();

  for (const group of samplerGroups) {
    group.countEl = el("span", { class: "count" });
    const details = el(
      "details",
      { class: "param-group", open: openState[group.name] ?? group.open },
      el("summary", {}, group.name, group.countEl),
      ...group.params.map(renderParam),
    );
    details.addEventListener("toggle", () => {
      openState[group.name] = details.open;
      save("groups-open", openState);
    });
    container.append(details);
  }
  updateGroupCounts();
}

async function loadSamplerSpecs() {
  const res = await fetch("samplers.json");
  samplerGroups = (await res.json()).groups;
  for (const group of samplerGroups) {
    for (const spec of group.params) paramSpecs.set(spec.key, spec);
  }
}

async function refreshOverrides() {
  try {
    const data = await (await api("/v1/sampling/overrides")).json();
    state.overrides = {
      ...data,
      effective: { ...(data.overrides || {}), ...(data.model_overrides || {}) },
    };
  } catch {
    state.overrides = null;
  }

  const select = $("preset-select");
  const presets = state.overrides?.presets || [];
  const selected = state.overrides?.selected_preset || "";
  select.replaceChildren(
    el("option", { value: "" }, "None"),
    ...presets.map((p) => el("option", { value: p }, p)),
  );
  if (selected && !presets.includes(selected)) select.append(el("option", { value: selected }, selected));
  select.value = selected;
  select.disabled = state.permission !== "admin";

  const info = [];
  if (state.overrides?.model_preset) info.push(`Model preset: ${state.overrides.model_preset}.`);
  const count = Object.keys(state.overrides?.effective || {}).length;
  if (count) info.push(`${count} server override${count === 1 ? "" : "s"} in effect; unset controls show them as defaults.`);
  if (state.permission === "admin") info.push("Changing the preset affects every client of this server.");
  $("preset-info").textContent = info.join(" ");

  renderParams();
}

async function switchPreset() {
  const preset = $("preset-select").value;
  try {
    if (preset) await api("/v1/sampling/override/switch", { method: "POST", body: { preset } });
    else await api("/v1/sampling/override/unload", { method: "POST" });
    setStatus(preset ? `Server preset switched to ${preset}` : "Server preset cleared");
  } catch (e) {
    setStatus(e.message, true);
  }
  await refreshOverrides();
}

// ---------------------------------------------------------------------------
// Models

// The model picked in the list, else the filter text taken as a path under
// the model directory
function listTarget() {
  return state.selectedModel || $("model-filter").value.trim() || null;
}

// A model_name in the load options, which the server resolves when parsing.
// Read approximately here, just to show the load target while typing.
function optionsModelName() {
  const text = $("load-options").value;
  const match = /^model_name:\s*["']?(.+?)["']?\s*(#.*)?$/m.exec(text) || /"model_name"\s*:\s*"([^"]+)"/.exec(text);
  return match ? match[1].trim() : null;
}

// What Load loads: a model_name in the load options, else the list target
function modelLoadName() {
  return optionsModelName() || listTarget();
}

function trimPath(name) {
  return name.replace(/\/+$/, "");
}

// The loaded model's full path ends with its name under the model directory
function matchesLoadedPath(name) {
  const path = state.modelPath;
  const n = trimPath(name);
  return !!path && !!n && (path === n || path.endsWith("/" + n));
}

// Whether a name from the list, the recent paths or the filter is the loaded
// model. A shorter name can match the end of the path by chance (a flat
// "4.0bpw" folder vs "<model>/exl3/4.0bpw"), so the longest listed match wins.
function isLoadedModel(name) {
  if (!state.currentModel) return false;
  if (!state.modelPath) return trimPath(name).split("/").pop() === state.currentModel.id;
  return matchesLoadedPath(name) && (!state.loadedName || trimPath(name).length >= state.loadedName.length);
}

// ---- Load options, pre-filled with the settings that apply to the target

const editedOptions = new Map();
let optionsTarget = null;
let optionsServerText = "";
let optionsTimer = null;
let optionsRequest = 0;

function setOptionsNote(text, isError = false) {
  const note = $("load-options-note");
  note.textContent = text ? `· ${text}` : "";
  note.classList.toggle("error", isError);
}

// Fill the load options for the current list target, keeping any edits made
// for the previous one in case it's picked again
function syncLoadOptions(force = false) {
  const name = listTarget();
  if (name === optionsTarget && !force) return;

  const box = $("load-options");
  if (optionsTarget && box.value !== optionsServerText) editedOptions.set(optionsTarget, box.value);
  if (force) editedOptions.delete(name);

  optionsTarget = name;
  optionsServerText = "";
  clearTimeout(optionsTimer);
  const request = ++optionsRequest;

  if (editedOptions.has(name)) {
    box.value = editedOptions.get(name);
    setOptionsNote("edited");
    updateModelButtons();
    return;
  }

  box.value = "";
  setOptionsNote("");
  updateModelButtons();
  if (!name || state.permission !== "admin") return;

  // Typed paths are looked up once typing pauses
  const delay = name === state.selectedModel ? 0 : 350;
  optionsTimer = setTimeout(async () => {
    try {
      const res = await api(`/v1/model/load_config?model_name=${encodeURIComponent(name)}`);
      const data = await res.json();
      if (request !== optionsRequest || box.value.trim()) return;
      optionsServerText = data.yaml;
      box.value = data.yaml;
      setOptionsNote(data.has_tabby_config ? "tabby_config.yml" : "");
      if (data.has_tabby_config) $("load-options-details").open = true;
    } catch (e) {
      // A path still being typed doesn't exist yet
      if (request === optionsRequest && e.status !== 404) setOptionsNote(e.message, true);
    }
    updateModelButtons();
  }, delay);
}

// The load options as a request body, parsed by the server so they can be YAML
async function parseLoadOptions() {
  const text = $("load-options").value;
  if (!text.trim()) return {};
  const res = await api("/v1/model/load_config/parse", { method: "POST", body: { yaml: text } });
  return (await res.json()).settings;
}

function rememberModel(name) {
  if (state.models.some((m) => m.id === name)) return;
  recentModels = [name, ...recentModels.filter((n) => n !== name)].slice(0, 8);
  save("recent-models", recentModels);
}

function updateModelButtons() {
  const admin = state.permission === "admin";
  const name = modelLoadName();
  const isLoaded = !!name && isLoadedModel(name);
  const loadButton = $("load-model");
  loadButton.textContent = isLoaded ? "Reload" : "Load";
  loadButton.disabled = !admin || state.loading || !name;
  $("load-target").textContent = admin && name ? `Load target: ${name}` + (optionsModelName() ? " (from load options)" : "") : "";
  $("unload-model").disabled = !admin || state.loading || !state.currentModel;
}

function renderModelInfo() {
  const info = $("model-info");
  const model = state.currentModel;
  const dot = $("model-dot");
  dot.className = "dot" + (state.loading ? " busy" : model ? " ok" : "");
  $("topbar-model-name").textContent = state.loading
    ? `Loading ${state.loading}…`
    : state.loadedName || model?.id || "No model loaded";

  if (!model) {
    info.replaceChildren(el("div", { class: "muted" }, "No model loaded"));
    return;
  }

  const p = model.parameters || {};
  const rows = [
    ["Context", p.max_seq_len],
    ["Cache", p.cache_size && `${p.cache_size} (${p.cache_mode})`],
    ["Batch", p.max_batch_size],
    ["Template", p.prompt_template],
    ["Vision", p.use_vision ? "yes" : null],
    ["Draft", p.draft?.id],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");

  info.replaceChildren(
    el("div", { class: "name" }, state.loadedName || model.id),
    el("dl", {}, rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, String(v))])),
  );
}

function toggleFavorite(name) {
  favoriteModels = favoriteModels.includes(name)
    ? favoriteModels.filter((n) => n !== name)
    : [...favoriteModels, name].sort((a, b) => a.localeCompare(b));
  save("favorite-models", favoriteModels);
  renderModelList();
}

function renderModelList() {
  const list = $("model-list");
  const filter = $("model-filter").value.trim().toLowerCase();
  const admin = state.permission === "admin";
  const byId = new Map(state.models.map((m) => [m.id, m]));
  const favorites = new Set(favoriteModels);

  // Favorites first (listed or not, so starred paths stay loadable), then
  // recently loaded paths the list doesn't have, then the rest of the list
  const models = [
    ...(admin ? favoriteModels.map((id) => byId.get(id) || { id, unlisted: true }) : []),
    ...(admin ? recentModels.filter((n) => !byId.has(n) && !favorites.has(n)).map((id) => ({ id, recent: true })) : []),
    ...state.models.filter((m) => !favorites.has(m.id)),
  ].filter((m) => !filter || m.id.toLowerCase().includes(filter));
  list.replaceChildren();

  for (const m of models) {
    const item = el(
      "li",
      {
        role: "option",
        class: (m.id === state.selectedModel ? "selected " : "") + (isLoadedModel(m.id) ? "loaded" : ""),
        "aria-selected": m.id === state.selectedModel ? "true" : "false",
        title: m.meta?.n_ctx_train ? `${m.id} (trained context ${m.meta.n_ctx_train})` : m.id,
        onclick: () => {
          state.selectedModel = state.selectedModel === m.id ? null : m.id;
          renderModelList();
          syncLoadOptions();
        },
        ondblclick: () => {
          state.selectedModel = m.id;
          syncLoadOptions();
          loadModel();
        },
      },
      el("span", { class: "dot" }),
      el("span", { class: "model-name" }, m.id),
      m.recent ? el("span", { class: "tag" }, "recent") : null,
      m.unlisted ? el("span", { class: "tag", title: "Not in the model list, loads as a path" }, "path") : null,
    );

    if (admin) {
      const starred = favorites.has(m.id);
      const star = el(
        "button",
        {
          type: "button",
          class: "star" + (starred ? " starred" : ""),
          title: starred ? "Remove from favorites" : "Add to favorites",
          "aria-label": starred ? `Remove ${m.id} from favorites` : `Add ${m.id} to favorites`,
          "aria-pressed": starred ? "true" : "false",
        },
        icon("star"),
      );
      // The row selects on click and loads on double click; the star does neither
      star.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleFavorite(m.id);
      });
      star.addEventListener("dblclick", (e) => e.stopPropagation());
      item.append(star);
    }
    list.append(item);
  }

  if (!models.length) {
    let text = "No models found";
    if (filter) text = "No match. Load takes the text above as a path under the model directory.";
    else if (state.permission && state.permission !== "admin") text = "An admin key is needed to list models.";
    list.append(el("li", { class: "empty" }, text));
  }
}

async function refreshModels() {
  try {
    state.currentModel = await (await api("/v1/model")).json();
  } catch {
    state.currentModel = null;
  }

  // The model card only names the folder; /props has the full path
  state.modelPath = null;
  if (state.currentModel) {
    try {
      state.modelPath = trimPath((await (await api("/props")).json()).model_path || "") || null;
    } catch {
      // Fall back to matching on the folder name
    }
  }

  try {
    const data = await (await api("/v1/model/list")).json();
    state.models = (data.data || []).sort((a, b) => a.id.localeCompare(b.id));
  } catch (e) {
    state.models = [];
    $("model-message").textContent = e.message;
  }

  const candidates = [...favoriteModels, ...recentModels, ...state.models.map((m) => m.id)].map(trimPath);
  state.loadedName = candidates.filter(matchesLoadedPath).sort((a, b) => b.length - a.length)[0] || null;

  renderModelInfo();
  renderModelList();
  updateModelButtons();
}

async function loadModel() {
  if (!modelLoadName() || state.loading) return;

  const message = $("model-message");
  let options;
  try {
    options = await parseLoadOptions();
  } catch (e) {
    message.textContent = e.message.startsWith("Load options") ? e.message : `Load options: ${e.message}`;
    message.classList.add("error");
    return;
  }

  const named = typeof options.model_name === "string" ? options.model_name.trim() : "";
  const name = named || listTarget();
  if (!name) return;

  const bar = $("load-progress");
  message.classList.remove("error");
  message.textContent = "";
  state.loading = name;
  bar.hidden = false;
  bar.firstElementChild.style.width = "0";
  renderModelInfo();
  updateModelButtons();

  try {
    // Loading the model that is already loaded is a no-op, so a reload with
    // different options needs an unload first
    if (isLoadedModel(name)) {
      message.textContent = "Unloading…";
      await api("/v1/model/unload", { method: "POST" });
    }

    const res = await api("/v1/model/load", { method: "POST", body: { ...options, model_name: name } });
    for await (const data of sseData(res)) {
      const event = JSON.parse(data);
      if (event.error) throw new Error(event.error.message);
      const label = event.model_type === "model" ? "model" : `${event.model_type} model`;
      message.textContent = `Loading ${label}: ${event.module} / ${event.modules} modules`;
      bar.firstElementChild.style.width = `${(100 * event.module) / event.modules}%`;
    }
    message.textContent = `Loaded ${name}`;
    rememberModel(name);
  } catch (e) {
    message.textContent = e.message;
    message.classList.add("error");
  } finally {
    state.loading = false;
    bar.hidden = true;
    await refreshModels();
    await refreshOverrides();
  }
}

async function unloadModel() {
  if (!state.currentModel || state.loading) return;
  const message = $("model-message");
  message.classList.remove("error");
  try {
    await api("/v1/model/unload", { method: "POST" });
    message.textContent = "Model unloaded";
  } catch (e) {
    message.textContent = e.message;
    message.classList.add("error");
  }
  await refreshModels();
  await refreshOverrides();
}

// ---------------------------------------------------------------------------
// Connection

async function checkConnection() {
  const pill = $("connection-state");
  try {
    const data = await (await api("/v1/auth/permission")).json();
    state.permission = data.permission;
    pill.textContent = data.permission;
    pill.className = "pill ok";
    return true;
  } catch (e) {
    state.permission = null;
    pill.textContent = e.status === 401 ? "no key" : "offline";
    pill.className = "pill bad";
    return e;
  }
}

function openConnectionDialog(message = "") {
  const dialog = $("connection-dialog");
  if (dialog.open) return;
  $("conn-url").value = settings.baseUrl;
  $("conn-key").value = settings.apiKey;
  $("conn-result").textContent = message;
  $("conn-result").classList.toggle("error", !!message);
  dialog.showModal();
}

async function connect() {
  const result = await checkConnection();
  if (result !== true) {
    openConnectionDialog(
      result.status === 401 ? "The server needs an API key." : `Could not reach the server: ${result.message}`,
    );
  } else if (state.permission !== "admin") {
    $("model-message").textContent = "This key can chat but not list or load models. Use the admin key for that.";
  } else {
    $("model-message").textContent = "";
  }
  await Promise.all([refreshModels(), refreshOverrides()]);
}

// ---------------------------------------------------------------------------
// Layout

function applyTheme() {
  if (settings.theme) document.documentElement.dataset.theme = settings.theme;
  else delete document.documentElement.dataset.theme;
  $("theme-label").textContent = `Theme: ${settings.theme || "system"}`;
}

function applyFormatting() {
  $("toggle-markdown").hidden = !markdown;
  $("markdown-label").textContent = `Replies: ${settings.markdown ? "formatted" : "raw text"}`;
}

function closeDrawers() {
  $("sidebar").classList.remove("open");
  $("panel").classList.remove("open");
  $("scrim").hidden = true;
}

function bindEvents() {
  $("new-chat").addEventListener("click", newChat);

  $("messages").addEventListener("click", async (e) => {
    const button = e.target.closest(".code-copy");
    if (!button) return;
    await copyText(button.closest(".code-block").querySelector("pre code").textContent.replace(/\n$/, ""));
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = "Copy"), 1200);
  });

  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });

  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });
  $("input").addEventListener("input", autosize);

  $("input").addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (!files.length) return;
    // Text that comes along (e.g. a file's path) is still pasted
    if (!e.clipboardData.types.includes("text/plain")) e.preventDefault();
    addAttachments(files);
  });

  $("attach").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", () => {
    addAttachments([...$("file-input").files]);
    $("file-input").value = "";
  });

  // Files can be dropped anywhere on the page. Without this the browser would
  // open a dropped file in place of the client.
  let dragDepth = 0;
  const carriesFiles = (e) => e.dataTransfer?.types.includes("Files");
  document.addEventListener("dragenter", (e) => {
    if (!carriesFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    $("drop-overlay").hidden = false;
  });
  document.addEventListener("dragover", (e) => {
    if (carriesFiles(e)) e.preventDefault();
  });
  document.addEventListener("dragleave", (e) => {
    if (carriesFiles(e) && --dragDepth <= 0) {
      dragDepth = 0;
      $("drop-overlay").hidden = true;
    }
  });
  document.addEventListener("drop", (e) => {
    if (!carriesFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    $("drop-overlay").hidden = true;
    addAttachments([...e.dataTransfer.files]);
  });

  $("image-viewer").addEventListener("click", () => $("image-viewer").close());

  $("stop").addEventListener("click", () => state.generating?.controller.abort());
  $("regenerate").addEventListener("click", regenerate);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.generating && !$("connection-dialog").open) state.generating.controller.abort();
  });

  $("system-prompt").addEventListener("input", () => {
    session.system = $("system-prompt").value;
    updateSystemPreview();
  });
  $("system-prompt").addEventListener("change", () => persistSession());

  $("refresh-models").addEventListener("click", () => {
    refreshModels();
    refreshOverrides();
  });
  $("model-filter").addEventListener("input", () => {
    // A selection the filter hides would otherwise beat the typed path
    const filter = $("model-filter").value.trim().toLowerCase();
    if (state.selectedModel && !state.selectedModel.toLowerCase().includes(filter)) state.selectedModel = null;
    renderModelList();
    syncLoadOptions();
  });
  $("load-options").addEventListener("input", () => {
    setOptionsNote($("load-options").value !== optionsServerText ? "edited" : "");
    updateModelButtons();
  });
  $("reset-load-options").addEventListener("click", () => syncLoadOptions(true));
  $("load-model").addEventListener("click", loadModel);
  $("unload-model").addEventListener("click", unloadModel);
  $("preset-select").addEventListener("change", switchPreset);

  $("reset-params").addEventListener("click", () => {
    params = {};
    save("params", params);
    renderParams();
  });

  $("open-connection").addEventListener("click", () => openConnectionDialog());
  $("conn-cancel").addEventListener("click", () => $("connection-dialog").close("cancel"));
  $("connection-form").addEventListener("submit", (e) => {
    e.preventDefault();
    settings.baseUrl = $("conn-url").value.trim();
    settings.apiKey = $("conn-key").value.trim();
    save("settings", settings);
    $("connection-dialog").close();
    connect();
  });

  $("toggle-markdown").addEventListener("click", () => {
    settings.markdown = !settings.markdown;
    save("settings", settings);
    applyFormatting();
    renderThread();
  });

  $("toggle-theme").addEventListener("click", () => {
    const order = ["", "light", "dark"];
    settings.theme = order[(order.indexOf(settings.theme) + 1) % order.length];
    save("settings", settings);
    applyTheme();
  });

  for (const button of document.querySelectorAll("[data-drawer]")) {
    button.addEventListener("click", () => {
      const drawer = $(button.dataset.drawer);
      const open = !drawer.classList.contains("open");
      closeDrawers();
      drawer.classList.toggle("open", open);
      $("scrim").hidden = !open;
    });
  }
  $("scrim").addEventListener("click", closeDrawers);

  // Another tab may have changed the chats; pick up the session list
  window.addEventListener("storage", (e) => {
    if (e.key === STORE + "sessions") {
      sessionIndex = load("sessions", []);
      renderSessionList();
    }
  });
}

// ---------------------------------------------------------------------------
// Startup

// The server prints a link to the client with the admin key in the URL
// fragment, which the browser never sends to the server. Take the key from
// it, then clear it from the address bar.
function takeKeyFromLink() {
  const key = new URLSearchParams(location.hash.slice(1)).get("key");
  if (!key) return;

  // The link is for the server that printed it, not one set up earlier
  settings.baseUrl = "";
  settings.apiKey = key;
  save("settings", settings);
  history.replaceState(null, "", location.pathname + location.search);
}

async function init() {
  takeKeyFromLink();
  applyTheme();
  applyFormatting();
  bindEvents();

  const current = load("current", null);
  if (!(current && openSession(current)) && !(sessionIndex.length && openSession(sessionIndex[0].id))) {
    session = createSession();
    save("current", session.id);
    renderSessionList();
    renderThread();
  }

  try {
    await loadSamplerSpecs();
  } catch (e) {
    setStatus(`Could not load sampler definitions: ${e.message}`, true);
  }
  renderParams();

  await connect();
  $("input").focus();
  collectImages();
}

init();
