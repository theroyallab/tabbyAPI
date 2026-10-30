// TabbyAPI chat client. Plain browser JavaScript with no build step: it calls
// the regular API (chat completions, model and sampler override endpoints) and
// keeps its chats and settings in the browser's local storage.

import "./effects.js";
import {
  $,
  STORE,
  MODEL_CHANGED_KEY,
  api,
  copyText,
  el,
  fmt,
  icon,
  load,
  newId,
  onStorageError,
  remove,
  save,
  settings,
  sseData,
  takeKeyFromLink,
} from "./shared.js";

// ---------------------------------------------------------------------------
// State

let params = load("params", {});
let sessionIndex = load("sessions", []);
let session = null;

const state = {
  permission: null,
  currentModel: null,
  // Full path of the loaded model, and its name in the model list if listed
  modelPath: null,
  loadedName: null,
  overrides: null,
  generating: null,
};

let samplerGroups = [];
const paramSpecs = new Map();

// ---------------------------------------------------------------------------
// Helpers

function setStatus(text, isError = false) {
  const line = $("status-line");
  line.textContent = text || "";
  line.classList.toggle("error", isError);
}

// ---------------------------------------------------------------------------
// Markdown

// highlight.js and KaTeX are loaded the first time a reply needs them, so a
// page without code or math never fetches them. Once one arrives, the replies
// on screen are formatted again with it.
const LIBRARIES = {
  highlight: {
    scripts: [
      "vendor/highlight.js/highlight.min.js",
      // Beyond the common languages the main file bundles
      ...["dockerfile", "powershell", "cmake", "latex"].map((l) => `vendor/highlight.js/languages/${l}.min.js`),
    ],
    styles: [],
  },
  katex: { scripts: ["vendor/katex/katex.min.js"], styles: ["vendor/katex/katex.min.css"] },
};
const libraryLoads = {};

function loadLibrary(name) {
  if (libraryLoads[name]) return;
  const { scripts, styles } = LIBRARIES[name];
  const load = (tag, attrs) =>
    new Promise((resolve, reject) => {
      const node = el(tag, attrs);
      node.addEventListener("load", resolve);
      node.addEventListener("error", () => reject(new Error(`Could not load ${attrs.src || attrs.href}`)));
      document.head.append(node);
    });
  libraryLoads[name] = (async () => {
    // In order: the language files register with the main highlight.js file
    for (const src of scripts) await load("script", { src });
    await Promise.all(styles.map((href) => load("link", { rel: "stylesheet", href })));
  })().then(reformatReplies, (e) => console.warn(e.message));
}

// The message each formatted reply on screen shows, to format it again
const formattedContent = new WeakMap();

function reformatReplies() {
  const stick = nearBottom();
  for (const node of document.querySelectorAll("#messages .content.markdown")) {
    const msg = formattedContent.get(node);
    if (msg) setContent(node, msg);
  }
  if (stick) scrollToBottom();
}

// Highlighted code by language and source. Replies re-render on every
// streamed token, so code that hasn't changed isn't highlighted again
const highlightedCode = new Map();

function highlightCode(code, lang) {
  lang = lang.toLowerCase();
  if (!lang) return "";
  const hljs = window.hljs;
  if (!hljs) {
    loadLibrary("highlight");
    return "";
  }
  if (!hljs.getLanguage(lang)) return "";

  const key = `${lang}\n${code}`;
  let html = highlightedCode.get(key);
  if (html === undefined) {
    try {
      html = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
    } catch {
      html = "";
    }
  } else highlightedCode.delete(key);
  highlightedCode.set(key, html);
  if (highlightedCode.size > 64) highlightedCode.delete(highlightedCode.keys().next().value);
  return html;
}

// markdown-it (vendor/markdown-it.min.js) with its safe defaults: HTML in a
// reply is shown as text, and javascript:, vbscript: and file: links are
// refused. Single newlines are kept as line breaks, as models tend to mean them.
// highlight.js output is escaped HTML, with spans marking the tokens.
const markdown = window.markdownit
  ? window.markdownit({ linkify: true, breaks: true, highlight: highlightCode })
  : null;

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

// Math, typeset by KaTeX: $…$ and \(…\) inline, $$…$$ and \[…\] displayed,
// either on lines of their own or within a paragraph. Single dollars follow
// Pandoc's rule so prices stay text: no space just inside either one, and no
// digit right after the closing one ("$5 and $10" isn't math).
const typesetMath = new Map();

function renderMath(tex, display, source) {
  const katex = window.katex;
  if (!katex) {
    loadLibrary("katex");
    return `<span class="math-pending">${markdown.utils.escapeHtml(source)}</span>`;
  }

  const key = `${display ? "D" : "I"}${tex}`;
  let html = typesetMath.get(key);
  if (html === undefined) {
    try {
      // Bad TeX comes out as its source in the error color, with the error as a tooltip.
      // Without trust, commands like \href and \includegraphics are refused
      html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
    } catch {
      html = `<span class="math-pending">${markdown.utils.escapeHtml(source)}</span>`;
    }
  } else typesetMath.delete(key);
  typesetMath.set(key, html);
  if (typesetMath.size > 256) typesetMath.delete(typesetMath.keys().next().value);
  return html;
}

const MATH_DELIMITERS = [
  ["\\(", "\\)", false],
  ["\\[", "\\]", true],
  ["$$", "$$", true],
  ["$", "$", false],
];

function mathInline(state, silent) {
  const src = state.src;
  const start = state.pos;
  const delimiters = MATH_DELIMITERS.find(([open]) => src.startsWith(open, start));
  if (!delimiters) return false;
  const [open, close, display] = delimiters;
  const from = start + open.length;

  let end;
  if (open === "$") {
    if (!src[from] || /[\s$]/.test(src[from])) return false;
    end = from;
    for (;;) {
      end = src.indexOf("$", end + 1);
      if (end < 0) return false;
      if (src[end - 1] === "\\") continue;
      if (!/\s/.test(src[end - 1]) && !/[0-9]/.test(src[end + 1] || "")) break;
      // Any other dollar means this one was a price: "$5 and $10"
      return false;
    }
    // Nor does math run into a code span: "$5 or `$x`"
    if (src.slice(from, end).includes("`")) return false;
  } else {
    end = src.indexOf(close, from);
    if (end < 0) return false;
  }

  const tex = src.slice(from, end);
  if (!tex.trim()) return false;
  if (!silent) {
    const token = state.push(display ? "math_display" : "math_inline", "", 0);
    token.content = tex;
    token.markup = open;
    token.info = close;
  }
  state.pos = end + close.length;
  return true;
}

// $$ or \[ starting a line, up to a line that ends with $$ or \]
function mathBlock(state, startLine, endLine, silent) {
  if (state.sCount[startLine] - state.blkIndent >= 4) return false;
  const lineText = (line) => state.src.slice(state.bMarks[line] + state.tShift[line], state.eMarks[line]);
  const first = lineText(startLine);
  const delimiters = MATH_DELIMITERS.find(([open, , display]) => display && first.startsWith(open));
  if (!delimiters) return false;
  const [open, close] = delimiters;

  const lines = [first.slice(open.length)];
  let line = startLine;
  for (;;) {
    const text = lines[lines.length - 1].trimEnd();
    if (text.endsWith(close) && !(line === startLine && text.length < close.length)) {
      lines[lines.length - 1] = text.slice(0, -close.length);
      break;
    }
    if (++line >= endLine) return false;
    lines.push(lineText(line));
  }
  // "$$a$$ and more" on one line is inline math in a paragraph, not a block
  if (line === startLine && first.trimEnd() !== `${open}${lines[0]}${close}`) return false;

  const tex = lines.join("\n");
  if (!tex.trim()) return false;
  if (silent) return true;
  state.line = line + 1;
  const token = state.push("math_block", "", 0);
  token.block = true;
  token.content = tex;
  token.markup = open;
  token.info = close;
  token.map = [startLine, state.line];
  return true;
}

if (markdown) {
  // Ahead of backslash escapes, which would turn \( into (
  markdown.inline.ruler.before("escape", "math_inline", mathInline);
  // Ahead of every other block rule: the lines of a formula can look like Markdown, and
  // an "=" or "-" line would otherwise make a heading of the lines above it
  markdown.block.ruler.before("table", "math_block", mathBlock, {
    alt: ["paragraph", "reference", "blockquote", "list"],
  });
  const source = (token) => `${token.markup}${token.content}${token.info}`;
  markdown.renderer.rules.math_inline = (tokens, idx) => renderMath(tokens[idx].content, false, source(tokens[idx]));
  markdown.renderer.rules.math_display = (tokens, idx) => renderMath(tokens[idx].content, true, source(tokens[idx]));
  markdown.renderer.rules.math_block = (tokens, idx) =>
    `<div class="math-block">${renderMath(tokens[idx].content, true, source(tokens[idx]))}</div>\n`;
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

// While a reply streams in, an unfinished SVG is previewed as far as it has
// got: the lenient HTML parser closes whatever is still open and drops a tag
// cut off midway, and the serializer turns the result back into an SVG file.
function partialSvg(source) {
  const svg = new DOMParser().parseFromString(source, "text/html").querySelector("svg");
  return svg ? new XMLSerializer().serializeToString(svg) : null;
}

// Live previews belong to the streaming reply's content node, one per SVG
// block, and outlive its re-renders so the picture doesn't flicker. Each is
// redrawn at most every LIVE_SVG_INTERVAL ms, and swapped in only once the
// new image has loaded.
const LIVE_SVG_INTERVAL = 250;
const liveSvgs = new WeakMap();

function liveSvgPreview(node, index) {
  let list = liveSvgs.get(node);
  if (!list) liveSvgs.set(node, (list = []));
  return (list[index] ??= {
    view: el("p", { class: "svg-pending" }, "Drawing preview…"),
    source: "",
    drawn: null,
    last: 0,
    timer: null,
    shown: 0,
    seq: 0,
  });
}

function drawLiveSvg(preview) {
  preview.timer = null;
  preview.last = performance.now();
  const svg = partialSvg(preview.source);
  if (!svg || svg === preview.drawn) return;
  preview.drawn = svg;

  const seq = ++preview.seq;
  const loader = new Image();
  loader.addEventListener("load", () => {
    // Loads can finish out of order; never go back to an older picture
    if (seq < preview.shown) return;
    preview.shown = seq;
    if (preview.view.tagName !== "IMG") {
      const image = el("img", { alt: "SVG preview" });
      preview.view.replaceWith(image);
      preview.view = image;
    }
    preview.view.src = loader.src;
  });
  loader.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
}

function updateLiveSvg(preview, source) {
  preview.source = source;
  if (!preview.timer) {
    const wait = Math.max(0, preview.last + LIVE_SVG_INTERVAL - performance.now());
    preview.timer = setTimeout(() => drawLiveSvg(preview), wait);
  }
}

function addSvgPreviews(node, streaming) {
  node.querySelectorAll(".svg-preview").forEach((frame, index) => {
    const source = frame.parentElement.querySelector("code")?.textContent || "";
    const live = streaming ? liveSvgPreview(node, index) : null;

    // A block still streaming in has no closing tag yet
    if (/<\/svg>\s*$/i.test(source)) {
      const image = svgPreview(source);
      if (image.dataset.failed) {
        frame.append(svgError());
      } else if (live?.shown && !image.complete) {
        // Keep the live picture up until the finished one is ready
        clearTimeout(live.timer);
        frame.append(live.view);
        image.addEventListener("load", () => live.view.replaceWith(image), { once: true });
      } else frame.append(image);
    } else if (live) {
      updateLiveSvg(live, source);
      frame.append(live.view);
    } else {
      // Generation stopped partway, so show what there is
      const svg = partialSvg(source);
      const image = svg && svgPreview(svg);
      if (image && !image.dataset.failed) frame.append(image);
      frame.append(el("p", { class: "svg-pending" }, "This SVG is incomplete."));
    }
  });
}

// Only replies are formatted; the raw setting shows exactly what the model wrote
function setContent(node, msg) {
  if (markdown && settings.markdown && msg.role === "assistant") {
    node.classList.add("markdown");
    formattedContent.set(node, msg);
    node.innerHTML = markdown.render(msg.content);
    addSvgPreviews(node, state.generating?.message === msg);
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
// Model

// Models are loaded and unloaded on the status page; the chat shows which one
// is loaded, and picks up a change made there through MODEL_CHANGED_KEY

function trimPath(name) {
  return name.replace(/\/+$/, "");
}

// The loaded model's full path ends with its name under the model directory
function matchesLoadedPath(name) {
  const path = state.modelPath;
  const n = trimPath(name);
  return !!path && !!n && (path === n || path.endsWith("/" + n));
}

function renderModelInfo() {
  const info = $("model-info");
  const model = state.currentModel;
  $("model-dot").className = "dot" + (model ? " ok" : "");
  $("topbar-model-name").textContent = state.loadedName || model?.id || "No model loaded";

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

async function refreshModels() {
  try {
    state.currentModel = await (await api("/v1/model")).json();
  } catch {
    state.currentModel = null;
  }

  // The model card only names the folder; /props has the full path, and the
  // model list (admin only) its name under the model directory
  state.modelPath = null;
  state.loadedName = null;
  if (state.currentModel) {
    try {
      state.modelPath = trimPath((await (await api("/props")).json()).model_path || "") || null;
    } catch {
      // Fall back to the folder name
    }
  }

  if (state.modelPath && state.permission === "admin") {
    let listed = [];
    try {
      listed = ((await (await api("/v1/model/list")).json()).data || []).map((m) => m.id);
    } catch {
      // Show the folder name
    }
    const candidates = [...load("favorite-models", []), ...load("recent-models", []), ...listed].map(trimPath);
    state.loadedName = candidates.filter(matchesLoadedPath).sort((a, b) => b.length - a.length)[0] || null;
  }

  renderModelInfo();
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
  }
  // Models are loaded on the status page, which shows only the live status
  // without the admin key
  $("status-link").textContent = state.permission === "admin" ? "Load models and view server status" : "View server status";
  await Promise.all([refreshModels(), refreshOverrides()]);
}

// ---------------------------------------------------------------------------
// Layout

function renderThemeMenu() {
  $("theme-select").replaceChildren(
    ...window.tabbyTheme.THEMES.map((t) => el("option", { value: t.id }, t.label)),
  );
}

function applyTheme() {
  window.tabbyTheme.apply(settings.theme);
  $("theme-select").value = settings.theme;
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

// The buttons at either end of the top bar hide and show the side panels, or
// open them as drawers on screens too narrow to fit them alongside the chat
const SIDE_PANELS = {
  sidebar: { setting: "hideSidebar", label: "chats", drawer: matchMedia("(max-width: 760px)") },
  panel: { setting: "hidePanel", label: "settings", drawer: matchMedia("(max-width: 1100px)") },
};

function toggleSidePanel(name) {
  const side = SIDE_PANELS[name];
  if (side.drawer.matches) {
    const drawer = $(name);
    const open = !drawer.classList.contains("open");
    closeDrawers();
    drawer.classList.toggle("open", open);
    $("scrim").hidden = !open;
  } else {
    settings[side.setting] = !settings[side.setting];
    save("settings", settings);
    applySidePanels();
  }
}

function applySidePanels() {
  for (const [name, side] of Object.entries(SIDE_PANELS)) {
    const hidden = !!settings[side.setting];
    document.documentElement.classList.toggle(`${name}-hidden`, hidden);
    const label = side.drawer.matches || hidden ? `Show ${side.label}` : `Hide ${side.label}`;
    const button = document.querySelector(`[data-drawer="${name}"]`);
    button.title = label;
    button.setAttribute("aria-label", label);
  }
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

  $("theme-select").addEventListener("change", () => {
    settings.theme = $("theme-select").value;
    save("settings", settings);
    applyTheme();
  });

  for (const button of document.querySelectorAll("[data-drawer]")) {
    button.addEventListener("click", () => toggleSidePanel(button.dataset.drawer));
  }
  for (const side of Object.values(SIDE_PANELS)) side.drawer.addEventListener("change", applySidePanels);
  $("scrim").addEventListener("click", closeDrawers);

  // Another tab may have changed the chats; pick up the session list. The
  // status page announces model loads and unloads the same way
  window.addEventListener("storage", (e) => {
    if (e.key === STORE + "sessions") {
      sessionIndex = load("sessions", []);
      renderSessionList();
    } else if (e.key === STORE + MODEL_CHANGED_KEY) {
      refreshModels();
      refreshOverrides();
    }
  });
}

// ---------------------------------------------------------------------------
// Startup

async function init() {
  onStorageError((e) => setStatus(`Could not save to browser storage (${e.name}). Delete some chats to free space.`, true));
  takeKeyFromLink();
  renderThemeMenu();
  applyTheme();
  applyFormatting();
  applySidePanels();
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
