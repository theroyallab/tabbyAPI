// TabbyAPI status page: live generation jobs, cache statistics and
// throughput, with a view into each generation and a way to cancel it, and
// model loading with a form for the load options. Plain browser JavaScript
// with no build step, like the chat client. Any API key shows the live
// status; the models panel, the inspector and cancelling need the admin key.

import {
  $,
  STORE,
  MODEL_CHANGED_KEY,
  announceModelChange,
  api,
  applyTheme,
  copyText,
  el,
  icon,
  load,
  save,
  settings,
  sseData,
  takeKeyFromLink,
} from "./shared.js";

const POLL_INTERVAL = 1000;
const HIDDEN_POLL_INTERVAL = 5000;
// The throughput charts show this many seconds
const HISTORY_SECONDS = 300;

const state = {
  permission: null,
  models: [],
  currentModel: null,
  // Full path of the loaded model, and its name in the model list if listed
  modelPath: null,
  loadedName: null,
  selectedModel: null,
  loading: false,
  // Set while a load runs the model's warmup, after the weights
  warmingUp: false,
  // Load option groups from /v1/model/load_options
  schema: null,
  draftModels: [],
  // The load options that apply to configTarget (/v1/model/load_config)
  config: null,
  configTarget: null,
  configError: null,
  status: null,
  // Throughput samples, {t, gen, prefill} in tokens per second
  history: [],
  lastCounters: null,
  // The job shown in the inspector: {id, text, length, finished, prompt}
  inspect: null,
};

// Shared with the chat client, which used to load models itself
let recentModels = load("recent-models", []);
let favoriteModels = load("favorite-models", []);
const view = Object.assign({ onlySet: false, descriptions: false, openGroups: {} }, load("status-view", {}));

// Changes made on this page, per model: { model: {...}, draft_model: {...} }.
// A null value clears an option the model would otherwise inherit, so the
// backend default applies.
const edits = new Map();

// ---------------------------------------------------------------------------
// Formatting

function fmtCount(n) {
  if (n === null || n === undefined) return "–";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return n.toLocaleString();
}

function fmtRate(r) {
  if (r === null || r === undefined) return "–";
  return r >= 100 ? Math.round(r).toLocaleString() : r.toFixed(1);
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

function fmtBytes(bytes) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

function pct(part, whole) {
  return whole ? `${Math.round((100 * part) / whole)}%` : "–";
}

function setMessage(text, isError = false) {
  const message = $("model-message");
  message.textContent = text || "";
  message.classList.toggle("error", isError);
}

// ---------------------------------------------------------------------------
// Models

// The model picked in the list, else the filter text taken as a path under
// the model directory
function listTarget() {
  return state.selectedModel || $("model-filter").value.trim() || null;
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

function rememberModel(name) {
  if (state.models.some((m) => m.id === name)) return;
  recentModels = [name, ...recentModels.filter((n) => n !== name)].slice(0, 8);
  save("recent-models", recentModels);
}

function toggleFavorite(name) {
  favoriteModels = favoriteModels.includes(name)
    ? favoriteModels.filter((n) => n !== name)
    : [...favoriteModels, name].sort((a, b) => a.localeCompare(b));
  save("favorite-models", favoriteModels);
  renderModelList();
}

function renderModelInfo() {
  const info = $("model-info");
  const model = state.currentModel;
  $("unload-model").disabled = state.permission !== "admin" || !!state.loading || !model;

  // The top bar names the model for keys that don't see the models panel
  const barModel = $("bar-model");
  barModel.firstElementChild.className = "dot" + (model ? " ok" : "");
  barModel.lastElementChild.textContent = model ? state.loadedName || model.id : "No model loaded";
  barModel.title = state.modelPath || "";

  if (state.loading) {
    const doing = state.warmingUp ? "Warming up" : "Loading";
    info.replaceChildren(el("div", { class: "name loading" }, el("span", { class: "dot busy" }), `${doing} ${state.loading}…`));
    return;
  }
  if (!model) {
    info.replaceChildren(el("div", { class: "muted" }, "No model loaded"));
    return;
  }

  const p = model.parameters || {};
  const rows = [
    ["Context", p.max_seq_len?.toLocaleString()],
    ["Cache", p.cache_size && `${p.cache_size.toLocaleString()} tokens (${p.cache_mode})`],
    ["Batch", p.max_batch_size],
    ["Chunk", p.chunk_size],
    ["Template", p.prompt_template],
    ["Vision", p.use_vision ? "yes" : null],
    ["Draft", p.draft?.id],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");

  info.replaceChildren(
    el("div", { class: "name" }, el("span", { class: "dot ok" }), state.loadedName || model.id),
    el("dl", {}, rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, String(v))])),
  );
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

    list.append(
      el(
        "li",
        {
          role: "option",
          class: (m.id === state.selectedModel ? "selected " : "") + (isLoadedModel(m.id) ? "loaded" : ""),
          "aria-selected": m.id === state.selectedModel ? "true" : "false",
          title: m.meta?.n_ctx_train ? `${m.id} (trained context ${m.meta.n_ctx_train})` : m.id,
          onclick: () => {
            state.selectedModel = state.selectedModel === m.id ? null : m.id;
            renderModelList();
            syncOptions();
          },
          ondblclick: () => {
            state.selectedModel = m.id;
            renderModelList();
            syncOptions();
            loadModel();
          },
        },
        el("span", { class: "dot" }),
        el("span", { class: "model-name" }, m.id),
        m.recent ? el("span", { class: "tag" }, "recent") : null,
        m.unlisted ? el("span", { class: "tag", title: "Not in the model list, loads as a path" }, "path") : null,
        star,
      ),
    );
  }

  if (!models.length) {
    let text = "No models found";
    if (filter) text = "No match. Load takes the text above as a path under the model directory.";
    else if (!admin) text = "The admin key is needed to list models.";
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

  if (state.permission === "admin") {
    try {
      const data = await (await api("/v1/model/list")).json();
      state.models = (data.data || []).sort((a, b) => a.id.localeCompare(b.id));
    } catch (e) {
      state.models = [];
      setMessage(e.message, true);
    }
  }

  const candidates = [...favoriteModels, ...recentModels, ...state.models.map((m) => m.id)].map(trimPath);
  state.loadedName = candidates.filter(matchesLoadedPath).sort((a, b) => b.length - a.length)[0] || null;

  // With nothing picked, the form shows the loaded model's options
  if (!state.selectedModel && !$("model-filter").value.trim() && state.loadedName) {
    state.selectedModel = state.loadedName;
  }

  renderModelInfo();
  renderModelList();
  syncOptions();
}

// ---------------------------------------------------------------------------
// Load options

const SOURCE_KINDS = {
  default: { label: "default", title: "Not set: the backend default applies" },
  config: { label: "config.yml", title: "From use_as_default in config.yml" },
  model: { label: "tabby_config.yml", title: "From the model's tabby_config.yml" },
  edited: { label: "changed here", title: "Changed on this page" },
  cleared: { label: "cleared here", title: "Cleared on this page, so the backend default applies" },
};

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

function editsFor(name) {
  if (!edits.has(name)) edits.set(name, { model: {}, draft_model: {} });
  return edits.get(name);
}

function inheritedOption(section, name) {
  const config = state.config;
  if (!config) return { value: null, source: null };
  const values = section === "model" ? config.settings : config.settings.draft_model || {};
  const sources = section === "model" ? config.sources : config.sources.draft_model || {};
  return { value: values[name] ?? null, source: typeof sources[name] === "string" ? sources[name] : null };
}

function optionState(section, name) {
  const inherited = inheritedOption(section, name);
  const changes = edits.get(state.configTarget)?.[section] || {};
  const edited = Object.hasOwn(changes, name);
  const value = edited ? changes[name] : inherited.value;

  let kind = "default";
  if (edited) kind = value === null ? "cleared" : "edited";
  else if (inherited.source === "tabby_config.yml") kind = "model";
  else if (inherited.source === "use_as_default") kind = "config";
  return { value, inherited, edited, kind };
}

function showValue(field, value) {
  if (value === null || value === undefined) return "auto";
  if (field.type === "bool") return value ? "on" : "off";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  if (typeof value === "object") return Object.keys(value).length ? JSON.stringify(value) : "none";
  return String(value);
}

function sourceTitle(field, st) {
  const kind = SOURCE_KINDS[st.kind];
  if (!st.edited) return kind.title;
  const was = st.inherited.source
    ? `${showValue(field, st.inherited.value)} from ${st.inherited.source === "use_as_default" ? "config.yml" : st.inherited.source}`
    : `the default, ${showValue(field, field.default)}`;
  return `${kind.title} (was ${was})`;
}

function setOption(section, name, value) {
  const target = state.configTarget;
  if (!target) return;
  const changes = editsFor(target)[section];
  if (same(value, inheritedOption(section, name).value)) delete changes[name];
  else changes[name] = value;
  refreshOptionRow(section, name);
}

function undoOption(section, name) {
  const target = state.configTarget;
  if (!target) return;
  delete editsFor(target)[section][name];
  refreshOptionRow(section, name);
}

// Controls by field type. Each calls onSet with the parsed value, or null to
// leave the option unset; input it can't parse is marked and not applied.

function boolControl(field, st, onSet) {
  // The default's value is in the tooltip, so the label fits a narrow column
  const choices = [
    [null, "Default", `Backend default: ${showValue(field, field.default)}`],
    [true, "On"],
    [false, "Off"],
  ];
  return el(
    "div",
    { class: "segmented", role: "radiogroup", "aria-label": field.name },
    choices.map(([value, label, title]) =>
      el(
        "button",
        {
          type: "button",
          role: "radio",
          class: st.value === value ? "on" : "",
          "aria-checked": st.value === value ? "true" : "false",
          title,
          onclick: () => onSet(value),
        },
        label,
      ),
    ),
  );
}

function textInput(id, field, value, parse, onSet, attrs = {}) {
  const input = el("input", {
    type: "text",
    id,
    value: value ?? "",
    placeholder: `Default: ${showValue(field, field.default)}`,
    spellcheck: "false",
    autocomplete: "off",
    ...attrs,
  });
  input.addEventListener("change", () => {
    const raw = input.value.trim();
    if (!raw) return onSet(null);
    const parsed = parse(raw);
    input.classList.toggle("invalid", parsed === undefined);
    if (parsed !== undefined) onSet(parsed);
  });
  return input;
}

function numberOrUndefined(raw, integer) {
  const n = Number(raw);
  if (!Number.isFinite(n) || (integer && !Number.isInteger(n))) return undefined;
  return n;
}

function optionControl(field, section, st, onSet) {
  const id = `opt-${section}-${field.name}`;
  const value = st.value;

  switch (field.type) {
    case "bool":
      return boolControl(field, st, onSet);

    case "int":
    case "float": {
      const input = el("input", {
        type: "number",
        id,
        value: value ?? "",
        placeholder: `Default: ${showValue(field, field.default)}`,
        step: field.step ?? (field.type === "int" ? 1 : "any"),
        min: field.min ?? undefined,
        max: field.max ?? undefined,
      });
      input.addEventListener("change", () => {
        const raw = input.value.trim();
        if (!raw) return onSet(null);
        const n = numberOrUndefined(raw, field.type === "int");
        input.classList.toggle("invalid", n === undefined);
        if (n !== undefined) onSet(n);
      });
      return input;
    }

    case "choice": {
      if (field.free || field.name === "draft_model_name") {
        const listId = `${id}-choices`;
        const choices = field.name === "draft_model_name" ? state.draftModels : field.choices || [];
        return el(
          "div",
          { class: "free-choice" },
          textInput(id, field, value, (raw) => raw, onSet, { list: listId }),
          el("datalist", { id: listId }, choices.map((c) => el("option", { value: c }))),
        );
      }
      const choices = [...(field.choices || [])];
      if (value !== null && !choices.includes(String(value))) choices.push(String(value));
      const select = el(
        "select",
        { id },
        el("option", { value: "" }, `Default (${showValue(field, field.default)})`),
        choices.map((c) => el("option", { value: c }, c)),
      );
      select.value = value ?? "";
      select.addEventListener("change", () => onSet(select.value || null));
      return select;
    }

    case "list":
      return textInput(
        id,
        field,
        Array.isArray(value) ? value.join(", ") : value,
        (raw) => {
          const items = raw.split(/[\s,]+/).filter(Boolean).map((item) => numberOrUndefined(item, false));
          return items.includes(undefined) ? undefined : items;
        },
        onSet,
      );

    case "number_or_auto":
      return textInput(id, field, value, (raw) => (raw.toLowerCase() === "auto" ? "auto" : numberOrUndefined(raw, false)), onSet);

    case "json": {
      const area = el("textarea", {
        id,
        rows: value && Object.keys(value).length > 1 ? 4 : 2,
        spellcheck: "false",
        placeholder: `Default: ${showValue(field, field.default)}. A JSON object, e.g. {"key": "value"}`,
      });
      area.value = value === null || value === undefined ? "" : JSON.stringify(value, null, 2);
      area.addEventListener("change", () => {
        const raw = area.value.trim();
        if (!raw) return onSet(null);
        let parsed;
        try {
          parsed = JSON.parse(raw);
        } catch {
          parsed = undefined;
        }
        const valid = parsed !== null && typeof parsed === "object" && !Array.isArray(parsed);
        area.classList.toggle("invalid", !valid);
        if (valid) onSet(parsed);
      });
      return area;
    }

    case "text":
      // layer_map: a JSON list of layer indices, or a string
      return textInput(
        id,
        field,
        value === null || typeof value === "string" ? value : JSON.stringify(value),
        (raw) => {
          if (!raw.startsWith("[")) return raw;
          try {
            return JSON.parse(raw);
          } catch {
            return undefined;
          }
        },
        onSet,
      );

    default:
      return textInput(id, field, value, (raw) => raw, onSet);
  }
}

function optionRow(section, field) {
  const st = optionState(section, field.name);
  const kind = SOURCE_KINDS[st.kind];
  const row = el(
    "div",
    { class: `opt opt-${st.kind}`, "data-key": `${section}.${field.name}` },
    el(
      "div",
      { class: "opt-head" },
      // Hovering shows the full name and the description (showOptionTip)
      el("label", { for: `opt-${section}-${field.name}`, "data-tip": `${section}.${field.name}` }, field.name),
      el("span", { class: `src src-${st.kind}`, title: sourceTitle(field, st) }, kind.label),
      el(
        "button",
        {
          type: "button",
          class: "opt-undo",
          hidden: !st.edited,
          title: "Undo this change",
          "aria-label": `Undo the change to ${field.name}`,
          onclick: () => undoOption(section, field.name),
        },
        icon("undo"),
      ),
    ),
    optionControl(field, section, st, (value) => setOption(section, field.name, value)),
    view.descriptions && field.description ? el("p", { class: "opt-desc" }, field.description) : null,
  );
  row.hidden = view.onlySet && st.kind === "default";
  return row;
}

function fieldByKey(key) {
  const [section, name] = key.split(".");
  for (const group of state.schema?.groups || []) {
    if (group.section !== section) continue;
    const field = group.fields.find((f) => f.name === name);
    if (field) return { group, field };
  }
  return null;
}

function refreshOptionRow(section, name) {
  const key = `${section}.${name}`;
  const found = fieldByKey(key);
  const old = $("options-form").querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (found && old) old.replaceWith(optionRow(section, found.field));
  renderGroupCounts();
  renderLegend();
  updateModelButtons();
}

function groupCounts(group) {
  let set = 0;
  let changed = 0;
  for (const field of group.fields) {
    const st = optionState(group.section, field.name);
    if (st.kind !== "default") set++;
    if (st.edited) changed++;
  }
  return { set, changed };
}

function renderGroupCounts() {
  for (const details of $("options-form").querySelectorAll("details.opt-group")) {
    const group = state.schema.groups.find((g) => g.name === details.dataset.group);
    const { set, changed } = groupCounts(group);
    details.querySelector(".count").textContent = set ? `${set} set` + (changed ? `, ${changed} changed` : "") : "";
  }
}

function renderLegend() {
  const counts = { default: 0, config: 0, model: 0, edited: 0, cleared: 0 };
  if (state.schema && state.configTarget) {
    for (const group of state.schema.groups) {
      for (const field of group.fields) counts[optionState(group.section, field.name).kind]++;
    }
  }
  const legend = $("options-legend");
  legend.hidden = !state.configTarget;
  legend.replaceChildren(
    ...["model", "config", "edited", "cleared", "default"]
      .filter((kind) => counts[kind] || kind !== "cleared")
      .map((kind) =>
        el("span", { class: `src src-${kind}`, title: SOURCE_KINDS[kind].title }, `${SOURCE_KINDS[kind].label} · ${counts[kind]}`),
      ),
  );
  $("reset-edits").hidden = !(counts.edited + counts.cleared);
}

function renderOptions() {
  const form = $("options-form");
  const note = $("options-note");
  const target = listTarget();
  $("options-target").textContent = target ? `for ${target}` : "";
  note.classList.remove("error");

  if (!state.schema || !target || state.configTarget !== target) {
    form.replaceChildren();
    if (state.permission !== "admin") note.textContent = "The admin key is needed to load models.";
    else if (!target) note.textContent = "Pick a model to see the options it loads with.";
    else note.textContent = state.configError || (state.schema ? "Reading the model's options…" : "");
    note.classList.toggle("error", !!state.configError);
    renderLegend();
    updateModelButtons();
    return;
  }

  const config = state.config;
  const notes = [];
  if (!config) {
    notes.push("Not a model folder in the model directory. Options set here apply to a load of this path.");
  } else if (config.has_tabby_config) {
    notes.push("The model's tabby_config.yml and config.yml's use_as_default fill in these options. Anything else gets the backend default.");
  } else {
    notes.push("This model has no tabby_config.yml: config.yml's use_as_default fills in options, and anything else gets the backend default.");
  }
  if (config?.ignored?.length) {
    notes.push(`tabby_config.yml also sets ${config.ignored.join(", ")}, which a load request can't set; those still apply.`);
  }
  note.textContent = notes.join(" ");

  form.replaceChildren(
    ...state.schema.groups.map((group) => {
      const { set } = groupCounts(group);
      const open = view.openGroups[group.name] ?? (set > 0 || group === state.schema.groups[0]);
      const details = el(
        "details",
        { class: "opt-group", "data-group": group.name, open },
        el("summary", {}, group.name, el("span", { class: "count" })),
        el("div", { class: "opt-grid" }, group.fields.map((field) => optionRow(group.section, field))),
      );
      details.addEventListener("toggle", () => {
        view.openGroups[group.name] = details.open;
        save("status-view", view);
      });
      return details;
    }),
  );
  renderGroupCounts();
  renderLegend();
  updateModelButtons();
}

// Fetch the options that apply to the list target, keeping changes made for
// other models in case they're picked again
let optionsTimer = null;
let optionsRequest = 0;

function syncOptions(force = false) {
  const name = listTarget();
  if (name === state.configTarget && !force) {
    renderOptions();
    return;
  }

  clearTimeout(optionsTimer);
  const request = ++optionsRequest;
  state.configError = null;
  if (state.configTarget !== name) {
    state.config = null;
    state.configTarget = null;
  }
  renderOptions();
  if (!name || state.permission !== "admin") return;

  // Typed paths are looked up once typing pauses
  const delay = name === state.selectedModel ? 0 : 350;
  optionsTimer = setTimeout(async () => {
    let config = null;
    try {
      config = await (await api(`/v1/model/load_config?model_name=${encodeURIComponent(name)}`)).json();
    } catch (e) {
      if (request !== optionsRequest) return;
      // A path that doesn't exist (yet) can still be edited and tried
      if (e.status !== 404) {
        state.configError = e.message;
        renderOptions();
        return;
      }
    }
    if (request !== optionsRequest) return;
    state.config = config;
    state.configTarget = name;
    renderOptions();
  }, delay);
}

// The options a load of the target gets: everything set, inherited or changed
function effectiveOptions() {
  // Options the form doesn't show (backend) keep the value the model inherits,
  // so a load or a save doesn't drop them
  const { draft_model: inheritedDraft, ...inherited } = state.config?.settings || {};
  const options = { ...inherited };
  const draft = { ...inheritedDraft };
  for (const group of state.schema?.groups || []) {
    const target = group.section === "model" ? options : draft;
    for (const field of group.fields) {
      const { value } = optionState(group.section, field.name);
      if (value === null || value === undefined) delete target[field.name];
      else target[field.name] = value;
    }
  }
  if (Object.keys(draft).length) options.draft_model = draft;
  return options;
}

function updateModelButtons() {
  const admin = state.permission === "admin";
  const name = listTarget();
  const ready = !!name && state.configTarget === name;
  const loadButton = $("load-model");
  loadButton.textContent = name && isLoadedModel(name) ? "Reload" : "Load";
  loadButton.disabled = !admin || !!state.loading || !ready;
  $("save-options").disabled = !admin || !ready || !state.config;

  const copied = load(COPIED_OPTIONS_KEY, null);
  $("copy-options").disabled = !ready;
  $("paste-options").disabled = !ready || !copied;
  $("paste-options").title = copied
    ? `Set these options to the ones copied from ${copied.model}`
    : "Copy a model's options first";
}

// ---- Copy and paste between models

// Kept in local storage, so a copy survives a reload and reaches other tabs
const COPIED_OPTIONS_KEY = "copied-load-options";

// Copy the options the form shows: what a load of this model would use
function copyOptions() {
  const name = state.configTarget;
  if (!name) return;
  save(COPIED_OPTIONS_KEY, { model: name, options: effectiveOptions() });
  setMessage(`Copied the load options of ${name}. Pick another model and paste them there.`);
  updateModelButtons();
}

// Make the form match the copied options: each option the copy sets gets its
// value, and the rest are cleared to the backend default. These are changes
// like any other, undone the same way, and nothing is loaded or saved
function pasteOptions() {
  const copied = load(COPIED_OPTIONS_KEY, null);
  const name = state.configTarget;
  if (!copied || !name || !state.schema) return;

  const changes = editsFor(name);
  for (const group of state.schema.groups) {
    const source = group.section === "model" ? copied.options : copied.options.draft_model || {};
    for (const field of group.fields) {
      const value = source[field.name] ?? null;
      if (same(value, inheritedOption(group.section, field.name).value)) delete changes[group.section][field.name];
      else changes[group.section][field.name] = value;
    }
  }
  renderOptions();
  setMessage(`Pasted the load options of ${copied.model}. Load, or save them to tabby_config.yml, to apply them.`);
}

async function loadModel() {
  const name = listTarget();
  if (!name || state.loading || state.configTarget !== name) return;

  const bar = $("load-progress");
  setMessage("");
  state.loading = name;
  bar.hidden = false;
  bar.firstElementChild.style.width = "0";
  renderModelInfo();
  updateModelButtons();

  try {
    // Loading the model that is already loaded is a no-op, so a reload with
    // different options needs an unload first
    if (isLoadedModel(name)) {
      setMessage("Unloading…");
      await api("/v1/model/unload", { method: "POST" });
      announceModelChange();
    }

    // The form shows every option that applies to the model, so it loads as
    // the complete set: an option cleared here gets the backend default
    // instead of coming back from tabby_config.yml or use_as_default
    const body = { inherit_defaults: false, ...effectiveOptions(), model_name: name };
    const res = await api("/v1/model/load", { method: "POST", body });
    for await (const data of sseData(res)) {
      const event = JSON.parse(data);
      if (event.error) throw new Error(event.error.message);
      // Warmup runs once the weights are loaded and counts steps, not modules
      const warmup = event.model_type === "warmup";
      if (warmup !== state.warmingUp) {
        state.warmingUp = warmup;
        renderModelInfo();
      }
      if (warmup) {
        setMessage(`Warming up: step ${event.module} / ${event.modules}`);
      } else {
        const label = event.model_type === "model" ? "model" : `${event.model_type} model`;
        setMessage(`Loading ${label}: ${event.module} / ${event.modules} modules`);
      }
      bar.firstElementChild.style.width = `${(100 * event.module) / event.modules}%`;
    }
    setMessage(`Loaded ${name}`);
    rememberModel(name);
  } catch (e) {
    setMessage(e.message, true);
  } finally {
    state.loading = false;
    state.warmingUp = false;
    bar.hidden = true;
    announceModelChange();
    await refreshModels();
  }
}

async function unloadModel() {
  if (!state.currentModel || state.loading) return;
  if (state.status?.jobs.length && !confirm("Unloading the model ends the requests in progress. Unload anyway?")) return;
  try {
    await api("/v1/model/unload", { method: "POST" });
    setMessage("Model unloaded");
  } catch (e) {
    setMessage(e.message, true);
  }
  announceModelChange();
  await refreshModels();
}

// Save the options as the target model's tabby_config.yml. The server leaves
// out whatever only repeats config.yml's use_as_default
async function saveOptions() {
  const name = listTarget();
  if (!name || state.configTarget !== name) return;
  const question =
    `Save these load options to the tabby_config.yml of ${name}?\n\n` +
    "Options that only repeat config.yml's use_as_default are left out, unless the file already has them. " +
    "An existing file is kept as tabby_config.bak.";
  if (!confirm(question)) return;

  const button = $("save-options");
  button.disabled = true;
  try {
    const res = await api("/v1/model/load_config/save", {
      method: "POST",
      body: { model_name: name, settings: effectiveOptions() },
    });
    const data = await res.json();
    const { draft_model: draft = {}, ...rest } = data.saved;
    const count = Object.keys(rest).length + Object.keys(draft).length;
    const backup = data.backup ? ` The previous file is kept as ${data.backup}.` : "";
    if (data.removed) {
      setMessage(`Removed the tabby_config.yml of ${name}, since every option matches use_as_default.${backup}`);
    } else if (!count) {
      setMessage("Nothing to save: every option matches use_as_default.");
    } else {
      setMessage(`Saved ${count} option${count === 1 ? "" : "s"} to the tabby_config.yml of ${name}.${backup}`);
    }

    // Show what applies to the model now
    edits.delete(name);
    syncOptions(true);
  } catch (e) {
    setMessage(e.message, true);
  } finally {
    updateModelButtons();
  }
}

async function loadSchema() {
  try {
    state.schema = await (await api("/v1/model/load_options")).json();
  } catch (e) {
    state.schema = null;
    state.configError = `Could not read the load options: ${e.message}`;
  }
  try {
    state.draftModels = ((await (await api("/v1/model/draft/list")).json()).data || []).map((m) => m.id).sort();
  } catch {
    state.draftModels = [];
  }
}

// ---------------------------------------------------------------------------
// Live status

const RUNNING_STAGES = new Set(["prefill", "generating", "resuming"]);

const STAGE_LABELS = {
  queued: "queued",
  prefill: "prefill",
  generating: "generating",
  waiting: "waiting",
  resuming: "resuming",
  finished: "finished",
};

let pollTimer = null;
let polling = false;

function setPollState(ok, message = "") {
  const pill = $("poll-state");
  pill.textContent = ok ? "live" : "paused";
  pill.className = "pill " + (ok ? "ok" : "bad");
  pill.title = ok ? `Updated every ${POLL_INTERVAL / 1000} s` : message || "Not connected";
}

async function poll() {
  // Overlapping polls would add samples a moment apart, which makes the rates jump
  if (polling) return;
  polling = true;
  clearTimeout(pollTimer);
  if (state.permission) {
    try {
      const data = await (await api("/v1/status")).json();
      const t = performance.now() / 1000;
      const last = state.lastCounters;
      if (last && t > last.t) {
        const dt = t - last.t;
        state.history.push({
          t,
          gen: Math.max(data.generated_tokens - last.generated, 0) / dt,
          prefill: Math.max(data.prefilled_tokens - last.prefilled, 0) / dt,
        });
        while (state.history.length && t - state.history[0].t > HISTORY_SECONDS) state.history.shift();
      }
      state.lastCounters = { t, generated: data.generated_tokens, prefilled: data.prefilled_tokens };

      // A model loaded or unloaded elsewhere (another tab, an inline load)
      if (!!data.cache !== !!state.currentModel && !state.loading) refreshModels();

      state.status = data;
      setPollState(true);
      renderLive();
      if (state.inspect) await pollInspect();
    } catch (e) {
      setPollState(false, e.message);
    }
  } else {
    setPollState(false, "Not connected");
  }
  polling = false;
  pollTimer = setTimeout(poll, document.hidden ? HIDDEN_POLL_INTERVAL : POLL_INTERVAL);
}

// The rate over the last few samples, which smooths out the poll jitter
function recentRate(key, samples = 3) {
  const tail = state.history.slice(-samples);
  return tail.length ? tail.reduce((sum, s) => sum + s[key], 0) / tail.length : null;
}

function renderTiles() {
  const data = state.status;
  const jobs = data?.jobs || [];
  const count = (pred) => jobs.filter(pred).length;
  const tiles = [
    ["Running", count((j) => RUNNING_STAGES.has(j.stage)), "Requests being processed"],
    ["Waiting", count((j) => j.stage === "waiting"), "Requeued requests waiting for room in the cache"],
    ["Queued", count((j) => j.stage === "queued"), "Requests not started yet"],
    ["Output", `${fmtRate(recentRate("gen"))} T/s`, "Tokens generated per second, all requests"],
    ["Prompt", `${fmtRate(recentRate("prefill"))} T/s`, "Prompt tokens processed per second, cached ones not counted"],
    ["Completed", fmtCount(data?.completed ?? 0), "Requests finished since the server started"],
    ["Generated", fmtCount(data?.generated_tokens ?? 0), "Tokens generated since the server started"],
  ];
  $("tiles").replaceChildren(
    ...tiles.map(([label, value, title]) =>
      el("div", { class: "tile", title }, el("div", { class: "tile-value" }, String(value)), el("div", { class: "tile-label" }, label)),
    ),
  );
}

function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function drawChart(canvas, key, color) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const now = performance.now() / 1000;
  const points = state.history.map((s) => [s.t, s[key]]);
  const peak = Math.max(1, ...points.map(([, v]) => v));
  // A round top for the scale
  const magnitude = 10 ** Math.floor(Math.log10(peak));
  const top = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((m) => m >= peak);

  const pad = { left: 44, right: 6, top: 6, bottom: 4 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const x = (t) => pad.left + plotW * (1 - (now - t) / HISTORY_SECONDS);
  const y = (v) => pad.top + plotH * (1 - v / top);

  ctx.font = `10px ${cssVar("--sans", "sans-serif")}`;
  ctx.fillStyle = cssVar("--fg-muted", "#888");
  ctx.strokeStyle = cssVar("--border", "rgba(0,0,0,0.1)");
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const fraction of [0, 0.5, 1]) {
    const v = top * fraction;
    ctx.beginPath();
    ctx.moveTo(pad.left, Math.round(y(v)) + 0.5);
    ctx.lineTo(width - pad.right, Math.round(y(v)) + 0.5);
    ctx.stroke();
    ctx.fillText(fmtRate(v), pad.left - 6, y(v));
  }

  if (points.length < 2) return;
  ctx.beginPath();
  points.forEach(([t, v], i) => (i ? ctx.lineTo(x(t), y(v)) : ctx.moveTo(x(t), y(v))));
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();

  ctx.lineTo(x(points.at(-1)[0]), y(0));
  ctx.lineTo(x(points[0][0]), y(0));
  ctx.closePath();
  ctx.globalAlpha = 0.15;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}

function renderCharts() {
  drawChart($("chart-gen"), "gen", cssVar("--chart-gen", "#c2410c"));
  drawChart($("chart-prefill"), "prefill", cssVar("--chart-prefill", "#2563eb"));
  $("chart-gen-now").textContent = `${fmtRate(recentRate("gen"))} T/s`;
  $("chart-prefill-now").textContent = `${fmtRate(recentRate("prefill"))} T/s`;
}

// A bar with segments [{value, class, title}] out of total
function meter(segments, total) {
  return el(
    "div",
    { class: "meter" },
    segments.map((s) =>
      el("div", { class: `meter-seg ${s.class}`, style: `width:${total ? (100 * s.value) / total : 0}%`, title: s.title }),
    ),
  );
}

function renderCache() {
  const cache = state.status?.cache;
  const box = $("cache-stats");
  if (!cache) {
    box.replaceChildren(el("p", { class: "muted" }, "No model loaded"));
    return;
  }

  const max = cache.max_tokens;
  const used = cache.used_tokens;
  const free = cache.free_tokens;
  const held = Math.max(max - used - free, 0);
  const rows = [
    el(
      "div",
      { class: "cache-row" },
      el("div", { class: "cache-head" }, el("strong", {}, "VRAM"), el("span", { class: "muted" }, `${fmtCount(max)} tokens`)),
      meter(
        [
          { value: used, class: "seg-used", title: `${used.toLocaleString()} tokens in pages active requests use` },
          { value: held, class: "seg-held", title: `${held.toLocaleString()} tokens of cached prompts no request uses right now` },
        ],
        max,
      ),
      el(
        "div",
        { class: "cache-detail" },
        el("span", {}, el("i", { class: "key seg-used" }), `in use ${fmtCount(used)} (${pct(used, max)})`),
        el("span", {}, el("i", { class: "key seg-held" }), `idle cached ${fmtCount(held)}`),
        el("span", {}, `empty ${fmtCount(free)}`),
        el("span", { title: "Tokens of complete prompt pages a later request could reuse, in use or not" }, `reusable ${fmtCount(cache.cached_tokens)}`),
      ),
    ),
  ];

  if (cache.tier_max_tokens) {
    const tier = cache.tier_cached_tokens;
    rows.push(
      el(
        "div",
        { class: "cache-row" },
        el("div", { class: "cache-head" }, el("strong", {}, "System memory"), el("span", { class: "muted" }, `${fmtCount(cache.tier_max_tokens)} tokens`)),
        meter([{ value: tier, class: "seg-tier", title: `${tier.toLocaleString()} reusable tokens held only in system memory` }], cache.tier_max_tokens),
        el("div", { class: "cache-detail" }, el("span", {}, `${fmtCount(tier)} more reusable tokens (${pct(tier, cache.tier_max_tokens)})`)),
      ),
    );
  }

  if (cache.disk_max_bytes) {
    rows.push(
      el(
        "div",
        { class: "cache-row" },
        el("div", { class: "cache-head" }, el("strong", {}, "Disk"), el("span", { class: "muted" }, `${fmtBytes(cache.disk_max_bytes)} for all models`)),
        meter(
          [{ value: cache.disk_used_bytes, class: "seg-disk", title: `${fmtBytes(cache.disk_used_bytes)} written for this model` }],
          cache.disk_max_bytes,
        ),
        el(
          "div",
          { class: "cache-detail" },
          el("span", {}, `${fmtBytes(cache.disk_used_bytes)} for this model`),
          el("span", {}, `${fmtCount(cache.disk_cached_tokens)} more reusable tokens`),
          el("span", {}, `${cache.disk_pages.toLocaleString()} pages, ${cache.disk_states.toLocaleString()} checkpoints`),
        ),
      ),
    );
  }

  if (cache.hit_rate !== null && cache.hit_rate !== undefined) {
    const sources = [];
    const tierRate = cache.tier_max_tokens ? cache.tier_hit_rate : null;
    const diskRate = cache.disk_max_bytes ? cache.disk_hit_rate : null;
    const vram = cache.hit_rate - (tierRate || 0);
    sources.push(`${Math.round(vram * 100)}% from VRAM`);
    if (tierRate !== null) sources.push(`${Math.round(Math.max(tierRate - (diskRate || 0), 0) * 100)}% from system memory`);
    if (diskRate !== null) sources.push(`${Math.round(diskRate * 100)}% from disk`);
    rows.push(
      el(
        "div",
        { class: "cache-row hit-rate", title: "The part of prompt pages served from the cache instead of being processed, since the model loaded" },
        el("strong", {}, `Hit rate ${Math.round(cache.hit_rate * 100)}%`),
        el("span", { class: "muted" }, sources.join(" · ")),
      ),
    );
  }
  box.replaceChildren(...rows);
}

function jobProgress(job) {
  if (job.stage === "prefill" || job.stage === "queued") {
    const total = Math.max(job.prompt_tokens, 1);
    return {
      fraction: job.prefill_tokens / total,
      text: `${job.prefill_tokens.toLocaleString()} / ${job.prompt_tokens.toLocaleString()} prompt` + (job.cached_tokens ? `, ${job.cached_tokens.toLocaleString()} cached` : ""),
    };
  }
  if (job.stage === "resuming") {
    return {
      fraction: job.resume_total ? job.resume_tokens / job.resume_total : 0,
      text: `${job.gen_tokens.toLocaleString()} tokens · restoring ${job.resume_tokens.toLocaleString()} / ${job.resume_total.toLocaleString()}`,
    };
  }
  return { fraction: null, text: `${job.gen_tokens.toLocaleString()} tokens · prompt ${job.prompt_tokens.toLocaleString()}` };
}

function jobRow(job, finished = false) {
  const progress = jobProgress(job);
  const speed = job.stage === "waiting" ? `waiting ${fmtDuration(job.waiting_for)}` : job.tokens_per_second ? `${fmtRate(job.tokens_per_second)} T/s` : "";
  const stage = finished ? job.finish_reason || "ended" : STAGE_LABELS[job.stage] || job.stage;
  const selected = state.inspect?.id === job.request_id;
  // Reading a request's text needs the admin key
  const admin = state.permission === "admin";

  return el(
    "div",
    {
      class: `job stage-${finished ? "finished" : job.stage}` + (selected ? " selected" : "") + (admin ? " inspectable" : ""),
      onclick: admin ? () => inspectJob(job.request_id) : undefined,
      title: admin ? "Show this request's text" : undefined,
    },
    el("div", { class: "job-label" }, job.label),
    el("span", { class: `stage-badge stage-${finished ? `finish-${stage}` : job.stage}` }, stage),
    el(
      "div",
      { class: "job-progress" },
      progress.fraction !== null && !finished
        ? el("div", { class: "progress mini" }, el("div", { style: `width:${Math.min(progress.fraction, 1) * 100}%` }))
        : null,
      el("span", { class: "muted" }, finished ? `${job.gen_tokens.toLocaleString()} tokens · prompt ${job.prompt_tokens.toLocaleString()}` : progress.text),
    ),
    el("span", { class: "job-speed" }, finished ? "" : speed),
    el("span", { class: "job-time muted" }, fmtDuration(job.duration)),
    el(
      "div",
      { class: "job-actions" },
      job.can_cancel
        ? el(
            "button",
            {
              type: "button",
              class: "icon-button small danger",
              title: "Cancel this request",
              "aria-label": `Cancel ${job.label}`,
              onclick: (e) => {
                e.stopPropagation();
                cancelJob(job);
              },
            },
            icon("stop"),
          )
        : null,
    ),
  );
}

function renderJobs() {
  const jobs = state.status?.jobs || [];
  $("jobs-count").textContent = jobs.length ? `${jobs.length} in progress` : "";
  $("jobs").replaceChildren(...(jobs.length ? jobs.map((j) => jobRow(j)) : [el("p", { class: "muted empty" }, "No requests in progress")]));

  const recent = state.status?.recent || [];
  $("recent").replaceChildren(...(recent.length ? recent.map((j) => jobRow(j, true)) : [el("p", { class: "muted empty" }, "Nothing yet")]));
}

function renderBackendVersion() {
  const backend = state.status?.backend;
  const label = $("backend-version");
  if (!backend?.version) {
    label.replaceChildren();
    return;
  }
  label.replaceChildren(
    `${backend.name} ${backend.version}`,
    backend.commit ? el("code", { title: "Running from a source checkout at this commit" }, ` · ${backend.commit}`) : null,
  );
}

function renderLive() {
  renderBackendVersion();
  renderTiles();
  renderCharts();
  renderCache();
  renderJobs();
}

async function cancelJob(job) {
  if (!confirm(`Cancel ${job.label}?\n\nThe client gets the text generated so far, with finish_reason "abort".`)) return;
  try {
    await api(`/v1/status/jobs/${encodeURIComponent(job.request_id)}/cancel`, { method: "POST" });
  } catch (e) {
    alert(e.message);
  }
  poll();
}

// ---------------------------------------------------------------------------
// Inspector

function findJob(id) {
  return state.status?.jobs.find((j) => j.request_id === id) || state.status?.recent.find((j) => j.request_id === id) || null;
}

function inspectJob(id) {
  if (state.inspect?.id === id || state.permission !== "admin") return;
  state.inspect = { id, text: "", length: 0, finished: false, prompt: null };
  $("inspector").hidden = false;
  $("inspect-text").textContent = "";
  $("inspect-prompt").textContent = "";
  $("inspect-prompt-details").open = false;
  renderInspector();
  renderJobs();
  pollInspect();
  $("inspector").scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function closeInspector() {
  state.inspect = null;
  $("inspector").hidden = true;
  renderJobs();
}

async function pollInspect() {
  const inspect = state.inspect;
  if (!inspect || inspect.finished) return;
  try {
    const data = await (await api(`/v1/status/jobs/${encodeURIComponent(inspect.id)}?offset=${inspect.length}`)).json();
    if (state.inspect !== inspect) return;

    const pane = $("inspect-text");
    const atBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
    if (data.offset === inspect.length && data.text) {
      inspect.text += data.text;
      pane.append(data.text);
    }
    inspect.length = data.text_length;
    inspect.finished = data.finished;
    if (atBottom) pane.scrollTop = pane.scrollHeight;
  } catch (e) {
    // Finished too long ago to be kept
    if (e.status === 404) inspect.finished = true;
    inspect.error = e.message;
  }
  renderInspector();
}

// The prompt is only fetched when its section is opened, since it can be long
async function loadPrompt() {
  const inspect = state.inspect;
  if (!inspect || inspect.prompt !== null) return;
  $("inspect-prompt").textContent = "Loading…";
  try {
    const data = await (await api(`/v1/status/jobs/${encodeURIComponent(inspect.id)}?offset=${inspect.length}&prompt=true`)).json();
    if (state.inspect !== inspect) return;
    inspect.prompt = data.prompt || "";
    $("inspect-prompt").textContent = inspect.prompt;
  } catch (e) {
    if (state.inspect === inspect) $("inspect-prompt").textContent = e.message;
  }
}

function renderInspector() {
  const inspect = state.inspect;
  if (!inspect) return;
  const job = findJob(inspect.id);
  $("inspect-label").textContent = job?.label || inspect.id;

  const parts = [];
  if (job) {
    const stage = job.stage === "finished" ? `finished (${job.finish_reason || "ended"})` : STAGE_LABELS[job.stage] || job.stage;
    parts.push(stage, `${job.gen_tokens.toLocaleString()} tokens`, `prompt ${job.prompt_tokens.toLocaleString()} tokens`, fmtDuration(job.duration));
  } else if (inspect.error) {
    parts.push(inspect.error);
  }
  if (!inspect.text && !inspect.finished) parts.push("no text yet");
  $("inspect-meta").textContent = parts.filter(Boolean).join(" · ");
  $("inspect-cancel").hidden = !job?.can_cancel;
}

// ---------------------------------------------------------------------------
// Connection

async function checkConnection() {
  const pill = $("connection-state");
  try {
    const data = await (await api("/v1/auth/permission")).json();
    state.permission = data.permission;
    pill.textContent = data.permission;
    pill.className = "pill" + (data.permission === "admin" ? " ok" : "");
    applyPermission();
    return true;
  } catch (e) {
    state.permission = null;
    pill.textContent = e.status === 401 ? "no key" : "offline";
    pill.className = "pill bad";
    applyPermission();
    return e;
  }
}

// Without the admin key the page shows only the live status: no models panel,
// and requests can't be inspected or cancelled
function applyPermission() {
  const admin = state.permission === "admin";
  document.body.classList.toggle("api-only", !admin);
  if (!admin && state.inspect) closeInspector();
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
    openConnectionDialog(result.status === 401 ? "The server needs an API key." : `Could not reach the server: ${result.message}`);
  }
  if (state.permission === "admin") await loadSchema();
  await refreshModels();
  poll();
}

// ---------------------------------------------------------------------------
// Startup

// The tooltip for an option's name: the name in full, since a long one is cut
// short in its column, then the description. A native title can't set the
// name apart
function showOptionTip(label) {
  const found = fieldByKey(label.dataset.tip);
  if (!found) return;
  const tip = $("option-tip");
  tip.replaceChildren(
    el("strong", {}, found.field.name),
    found.field.description ? el("div", {}, found.field.description) : null,
  );
  tip.hidden = false;

  // Below the name, or above it near the bottom of the window, kept on screen
  const rect = label.getBoundingClientRect();
  const width = tip.offsetWidth;
  const height = tip.offsetHeight;
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
  const below = rect.bottom + 6;
  tip.style.left = `${left}px`;
  tip.style.top = `${below + height > window.innerHeight - 8 ? rect.top - height - 6 : below}px`;
}

function hideOptionTip() {
  $("option-tip").hidden = true;
}

function bindEvents() {
  $("options-form").addEventListener("mouseover", (e) => {
    const label = e.target.closest("label[data-tip]");
    if (label) showOptionTip(label);
  });
  $("options-form").addEventListener("mouseout", (e) => {
    if (e.target.closest("label[data-tip]")) hideOptionTip();
  });
  $("models-pane").addEventListener("scroll", hideOptionTip, { passive: true });

  $("refresh-models").addEventListener("click", refreshModels);
  $("model-filter").addEventListener("input", () => {
    // A selection the filter hides would otherwise beat the typed path
    const filter = $("model-filter").value.trim().toLowerCase();
    if (state.selectedModel && !state.selectedModel.toLowerCase().includes(filter)) state.selectedModel = null;
    renderModelList();
    syncOptions();
  });
  $("load-model").addEventListener("click", loadModel);
  $("unload-model").addEventListener("click", unloadModel);
  $("save-options").addEventListener("click", saveOptions);
  $("copy-options").addEventListener("click", copyOptions);
  $("paste-options").addEventListener("click", pasteOptions);
  $("reset-edits").addEventListener("click", () => {
    edits.delete(state.configTarget);
    renderOptions();
  });

  $("only-set").checked = view.onlySet;
  $("only-set").addEventListener("change", () => {
    view.onlySet = $("only-set").checked;
    save("status-view", view);
    renderOptions();
  });
  $("show-descriptions").checked = view.descriptions;
  $("show-descriptions").addEventListener("change", () => {
    view.descriptions = $("show-descriptions").checked;
    save("status-view", view);
    renderOptions();
  });

  $("inspect-close").addEventListener("click", closeInspector);
  $("inspect-copy").addEventListener("click", async () => {
    await copyText(state.inspect?.text || "");
    $("inspect-copy").textContent = "Copied";
    setTimeout(() => ($("inspect-copy").textContent = "Copy"), 1200);
  });
  $("inspect-cancel").addEventListener("click", () => {
    const job = findJob(state.inspect?.id);
    if (job) cancelJob(job);
  });
  $("inspect-prompt-details").addEventListener("toggle", () => {
    if ($("inspect-prompt-details").open) loadPrompt();
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

  $("theme-select").replaceChildren(...window.tabbyTheme.THEMES.map((t) => el("option", { value: t.id }, t.label)));
  $("theme-select").value = settings.theme;
  $("theme-select").addEventListener("change", () => {
    settings.theme = $("theme-select").value;
    save("settings", settings);
    applyTheme();
    renderCharts();
  });

  // Another tab may have loaded a model, or changed the connection or theme
  window.addEventListener("storage", (e) => {
    if (e.key === STORE + MODEL_CHANGED_KEY) refreshModels();
    if (e.key === STORE + COPIED_OPTIONS_KEY) updateModelButtons();
    if (e.key === STORE + "settings") {
      Object.assign(settings, load("settings", {}));
      applyTheme();
      $("theme-select").value = settings.theme;
    }
  });
  window.addEventListener("resize", renderCharts);
  document.addEventListener("visibilitychange", () => !document.hidden && poll());
}

async function init() {
  takeKeyFromLink();
  applyTheme();
  bindEvents();
  renderModelInfo();
  renderLive();
  await connect();
}

init();
