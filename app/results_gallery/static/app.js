const PAGE_SIZE = 6;
const MAX_COMMENT = 2000;
const STATUS_LABEL = { ok: "Has result", missing: "Missing", error: "Error" };

// Lucide icons (ISC): https://lucide.dev
const LUCIDE = {
  eye:
    '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>',
  "eye-off":
    '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/>',
  "thumbs-up":
    '<path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>',
  "thumbs-down":
    '<path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/>',
  "message-square": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  "refresh-cw":
    '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
};

const state = {
  catalog: null,
  feedback: {},
  phase: "VIBE",
  model: "",
  run: "",
  operation: "",
  caseKey: "",
  status: "",
  query: "",
  page: 0,
  open: -1,
  showGt: {},
  commentOpen: "",
};

const $ = (id) => document.getElementById(id);

function lucide(name, size) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size || 16));
  svg.setAttribute("height", String(size || 16));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = LUCIDE[name] || "";
  return svg;
}

function setIcon(el, name, size) {
  el.replaceChildren(lucide(name, size));
}

function caseKeyOf(item) {
  return item.operation + "/" + item.case;
}

function modelView(item) {
  const models = item.models || {};
  return models[state.model] || { output: null, output_v: null, status: "missing", error: "" };
}

function showingGt(item) {
  return item.phase === "Prior" && item.has_gt && Boolean(state.showGt[item.key]);
}

function feedbackKey(item) {
  return [state.run || "", state.model, item.phase, item.operation, item.case || "", item.id].join("|");
}

function feedbackOf(item) {
  const row = state.feedback[feedbackKey(item)];
  return {
    vote: row && row.vote ? row.vote : "",
    comment: row && row.comment ? row.comment : "",
  };
}

function filtered() {
  const items = state.catalog ? state.catalog.items : [];
  const query = state.query.trim().toLowerCase();
  return items.filter((item) => {
    if (item.phase !== state.phase) return false;
    if (state.operation && item.operation !== state.operation) return false;
    if (state.phase === "Prior" && state.caseKey && caseKeyOf(item) !== state.caseKey) return false;
    const view = modelView(item);
    if (state.status && view.status !== state.status) return false;
    if (!query) return true;
    return [item.id, item.prompt, item.case_label, item.operation_label, item.edit_type, item.source_id]
      .join("\n")
      .toLowerCase()
      .includes(query);
  });
}

function imageUrl(kind, path, version, edge) {
  const encoded = String(path)
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  const params = new URLSearchParams();
  if (version) params.set("v", String(version));
  if (kind === "thumb") params.set("e", String(edge || 640));
  const query = params.toString();
  return "/" + kind + "/" + encoded + (query ? "?" + query : "");
}

function missingCount(items) {
  return items.filter((item) => modelView(item).status === "missing").length;
}

function fillSelect(select, options, value, placeholder) {
  select.replaceChildren();
  if (!options.length && placeholder) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = placeholder;
    select.append(empty);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  options.forEach((option) => {
    const node = document.createElement("option");
    node.value = option.id;
    node.textContent = option.label;
    select.append(node);
  });
  if (value && options.some((option) => option.id === value)) {
    select.value = value;
  } else if (options.length) {
    select.value = options[0].id;
  }
}

function renderFilters() {
  const catalog = state.catalog;
  if (!catalog) return;

  fillSelect(
    $("phase"),
    (catalog.phases || ["VIBE", "Prior"]).map((phase) => ({ id: phase, label: phase })),
    state.phase,
  );
  fillSelect(
    $("model"),
    (catalog.models || []).map((model) => ({ id: model.id, label: model.label || model.id })),
    state.model,
    "No model",
  );
  fillSelect(
    $("run"),
    (catalog.runs || []).map((run) => ({ id: run, label: run })),
    catalog.run || state.run,
    "No run",
  );

  const phaseItems = (catalog.items || []).filter((item) => item.phase === state.phase);
  const phaseOps = (catalog.operations || []).filter((item) => item.phase === state.phase);
  const opOptions = [{ id: "", label: "All operations" }];
  phaseOps.forEach((operation) => {
    const group = phaseItems.filter((item) => item.operation === operation.id);
    const missing = missingCount(group);
    const extra = missing ? " · " + missing + " missing" : "";
    opOptions.push({
      id: operation.id,
      label: operation.label + " (" + group.length + ")" + extra,
    });
  });
  fillSelect($("operation"), opOptions, state.operation);

  const caseField = $("case-field");
  const caseSelect = $("case");
  caseField.hidden = state.phase !== "Prior";
  if (state.phase !== "Prior") {
    fillSelect(caseSelect, [{ id: "", label: "All cases" }], "");
    caseSelect.disabled = true;
    return;
  }

  const selected = phaseOps.find((item) => item.id === state.operation);
  if (!selected) {
    fillSelect(caseSelect, [{ id: "", label: "Select an operation" }], "");
    caseSelect.disabled = true;
    return;
  }

  const caseOptions = [{ id: "", label: "All cases" }];
  (catalog.cases || [])
    .filter((item) => item.phase === "Prior" && item.operation === state.operation)
    .forEach((item) => {
      const key = item.operation + "/" + item.id;
      const group = phaseItems.filter((row) => row.operation === item.operation && row.case === item.id);
      const missing = missingCount(group);
      const extra = missing ? " · " + missing + " missing" : "";
      caseOptions.push({
        id: key,
        label: item.label + " (" + group.length + ")" + extra,
      });
    });
  fillSelect(caseSelect, caseOptions, state.caseKey);
}

function renderWarnings() {
  const warnings = (state.catalog && state.catalog.warnings) || [];
  const box = $("warnings");
  if (!warnings.length) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }
  const summary = document.createElement("summary");
  summary.textContent = warnings.length + (warnings.length === 1 ? " warning" : " warnings");
  const list = document.createElement("ul");
  warnings.forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    list.append(item);
  });
  box.hidden = false;
  box.replaceChildren(summary, list);
}

function summaryText(items) {
  if (!state.catalog) return "Loading…";
  const total = items.length;
  const phaseTotal = (state.catalog.items || []).filter((item) => item.phase === state.phase).length;
  const missing = missingCount(items);
  const errors = items.filter((item) => modelView(item).status === "error").length;
  const shown = total === phaseTotal ? total + " samples" : total + " / " + phaseTotal + " samples";
  const bits = [shown];
  if (missing) bits.push(missing + " missing");
  if (errors) bits.push(errors + " errors");
  const run = state.catalog.run ? " · " + state.catalog.run : "";
  return bits.join(" · ") + run;
}

function renderContext() {
  const line = $("context");
  if (!state.catalog || state.phase !== "Prior" || !state.caseKey) {
    line.hidden = true;
    line.textContent = "";
    return;
  }
  const found = state.catalog.cases.find((item) => item.operation + "/" + item.id === state.caseKey);
  const text = found ? found.description : "";
  line.hidden = !text;
  line.textContent = text;
}

function kickerText(item) {
  const parts = [item.phase, item.operation_label];
  if (item.case_label) parts.push(item.case_label);
  parts.push(item.id);
  return parts.join(" · ");
}

function resultTarget(item) {
  const view = modelView(item);
  if (showingGt(item)) {
    return { label: "Ground truth", path: item.gt, version: item.gt_v };
  }
  return { label: "Output", path: view.output, version: view.output_v };
}

function fillFrame(frame, label, path, version, kind) {
  frame.replaceChildren();
  if (!path) return;
  const image = document.createElement("img");
  image.alt = label;
  if (kind === "thumb") image.loading = "lazy";
  image.src = imageUrl(kind, path, version, 640);
  frame.append(image);
}

function eyeButton(item) {
  const on = showingGt(item);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "gt-toggle" + (on ? " on" : "");
  button.setAttribute("aria-label", on ? "Hide ground truth" : "Show ground truth");
  button.setAttribute("aria-pressed", on ? "true" : "false");
  button.append(lucide(on ? "eye-off" : "eye", 16));
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    toggleGt(item.key);
  });
  return button;
}

function iconButton(name, className, label, on, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "icon-btn" + (className ? " " + className : "") + (on ? " on" : "");
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-pressed", on ? "true" : "false");
  button.append(lucide(name, 16));
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    onClick();
  });
  return button;
}

function commentBox(item) {
  const fb = feedbackOf(item);
  const box = document.createElement("textarea");
  box.className = "comment-box";
  box.placeholder = "Add a comment";
  box.maxLength = MAX_COMMENT;
  box.value = fb.comment;
  box.addEventListener("click", (event) => event.stopPropagation());
  box.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      box.blur();
    }
  });
  box.addEventListener("input", () => {
    writeFeedbackLocal(item, { comment: box.value });
  });
  box.addEventListener("blur", () => {
    saveFeedback(item).catch(showError);
  });
  return box;
}

function feedbackBar(item) {
  const fb = feedbackOf(item);
  const key = feedbackKey(item);
  const wrap = document.createElement("div");
  wrap.addEventListener("click", (event) => event.stopPropagation());
  const row = document.createElement("div");
  row.className = "feedback";
  row.append(
    iconButton("thumbs-up", "like", "Like", fb.vote === "like", () => setVote(item, "like")),
    iconButton("thumbs-down", "dislike", "Dislike", fb.vote === "dislike", () => setVote(item, "dislike")),
    iconButton("message-square", "comment", "Comment", Boolean(fb.comment) || state.commentOpen === key, () => {
      state.commentOpen = state.commentOpen === key ? "" : key;
      render();
    }),
  );
  wrap.append(row);
  if (state.commentOpen === key) wrap.append(commentBox(item));
  return wrap;
}

function pane(label, path, version, kind) {
  const wrap = document.createElement("div");
  wrap.className = "pane";
  const caption = document.createElement("p");
  caption.className = "pane-label";
  caption.textContent = label;
  const holder = document.createElement("div");
  holder.className = "frame-wrap";
  const frame = document.createElement("div");
  frame.className = "frame";
  fillFrame(frame, label, path, version, kind);
  holder.append(frame);
  wrap.append(holder, caption);
  return { wrap, holder };
}

function card(item, index) {
  const view = modelView(item);
  const fb = feedbackOf(item);
  const article = document.createElement("article");
  article.className =
    "card" +
    (view.status === "error" ? " error" : "") +
    (fb.vote === "like" ? " liked" : "") +
    (fb.vote === "dislike" ? " disliked" : "");
  article.tabIndex = 0;

  const top = document.createElement("div");
  top.className = "card-top";
  const kicker = document.createElement("p");
  kicker.className = "kicker";
  kicker.textContent = kickerText(item);
  const badge = document.createElement("span");
  badge.className = "badge " + view.status;
  badge.textContent = STATUS_LABEL[view.status] || view.status;
  top.append(kicker, badge);

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = item.prompt || "(no prompt)";
  prompt.title = item.prompt || "";

  const pair = document.createElement("div");
  pair.className = "pair";
  const inputPane = pane("Input", item.input, item.input_v, "thumb");
  inputPane.wrap.append(feedbackBar(item));
  const target = resultTarget(item);
  const resultPane = pane(target.label, target.path, target.version, "thumb");
  if (item.phase === "Prior" && item.has_gt) {
    resultPane.holder.append(eyeButton(item));
  }
  pair.append(inputPane.wrap, resultPane.wrap);

  article.append(top, prompt, pair);
  const open = () => {
    state.open = index;
    render();
  };
  article.addEventListener("click", open);
  article.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  });
  return article;
}

function renderPager(total, pages) {
  const nav = $("pager");
  if (total <= PAGE_SIZE) {
    nav.hidden = true;
    nav.replaceChildren();
    return;
  }
  const prev = document.createElement("button");
  prev.type = "button";
  prev.textContent = "Prev";
  prev.disabled = state.page <= 0;
  prev.addEventListener("click", () => {
    state.page -= 1;
    state.open = -1;
    render();
  });
  const label = document.createElement("span");
  label.textContent = state.page + 1 + " / " + pages;
  const next = document.createElement("button");
  next.type = "button";
  next.textContent = "Next";
  next.disabled = state.page >= pages - 1;
  next.addEventListener("click", () => {
    state.page += 1;
    state.open = -1;
    render();
  });
  nav.hidden = false;
  nav.replaceChildren(prev, label, next);
}

function paintVoteButton(button, className, on) {
  button.className = "icon-btn " + className + (on ? " on" : "");
  button.setAttribute("aria-pressed", on ? "true" : "false");
}

function renderViewer(items) {
  const modal = $("viewer");
  const open = state.open >= 0 && state.open < items.length;
  modal.hidden = !open;
  document.body.classList.toggle("modal-open", open);
  if (!open) return;

  const item = items[state.open];
  const view = modelView(item);
  const target = resultTarget(item);
  const fb = feedbackOf(item);
  const key = feedbackKey(item);
  $("viewer-kicker").textContent = kickerText(item);
  $("viewer-title").textContent = item.id;
  $("viewer-prompt").textContent = item.prompt || "(no prompt)";
  const error = $("viewer-error");
  const message = view.error || "";
  error.hidden = !message;
  error.textContent = message;
  fillFrame($("viewer-input"), "Input " + item.id, item.input, item.input_v, "media");
  $("viewer-result-label").textContent = target.label;
  fillFrame($("viewer-output"), target.label + " " + item.id, target.path, target.version, "media");

  const gt = $("viewer-gt");
  const canGt = item.phase === "Prior" && item.has_gt;
  gt.hidden = !canGt;
  if (canGt) {
    const on = showingGt(item);
    gt.className = "gt-toggle" + (on ? " on" : "");
    gt.setAttribute("aria-pressed", on ? "true" : "false");
    gt.setAttribute("aria-label", on ? "Hide ground truth" : "Show ground truth");
    setIcon(gt, on ? "eye-off" : "eye", 16);
  }

  paintVoteButton($("viewer-like"), "like", fb.vote === "like");
  paintVoteButton($("viewer-dislike"), "dislike", fb.vote === "dislike");
  paintVoteButton($("viewer-comment-btn"), "comment", Boolean(fb.comment) || state.commentOpen === key);

  const box = $("viewer-comment");
  const showComment = state.commentOpen === key;
  box.hidden = !showComment;
  if (showComment && document.activeElement !== box) box.value = fb.comment;

  $("viewer-pos").textContent = state.open + 1 + " / " + items.length + " · arrow keys to move";
  $("viewer-prev").disabled = state.open <= 0;
  $("viewer-next").disabled = state.open >= items.length - 1;
}

function render() {
  const items = filtered();
  const pages = Math.max(1, Math.ceil(items.length / PAGE_SIZE) || 1);
  if (state.page > pages - 1) state.page = pages - 1;
  if (state.open >= items.length) state.open = -1;

  $("summary").textContent = summaryText(items);
  renderContext();

  const slice = items.slice(state.page * PAGE_SIZE, state.page * PAGE_SIZE + PAGE_SIZE);
  if (!slice.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = state.catalog ? "No samples match these filters." : "Loading…";
    $("grid").replaceChildren(empty);
  } else {
    $("grid").replaceChildren(...slice.map((item, index) => card(item, state.page * PAGE_SIZE + index)));
  }
  renderPager(items.length, pages);
  renderViewer(items);
}

function toggleGt(key) {
  state.showGt[key] = !state.showGt[key];
  render();
}

function writeFeedbackLocal(item, patch) {
  const key = feedbackKey(item);
  const prev = feedbackOf(item);
  const next = {
    vote: patch.vote !== undefined ? patch.vote : prev.vote,
    comment: patch.comment !== undefined ? String(patch.comment).slice(0, MAX_COMMENT) : prev.comment,
  };
  if (!next.vote && !next.comment) delete state.feedback[key];
  else state.feedback[key] = next;
  return next;
}

async function saveFeedback(item) {
  const next = feedbackOf(item);
  const response = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run: state.run || "",
      model: state.model,
      phase: item.phase,
      operation: item.operation,
      case: item.case || "",
      id: item.id,
      vote: next.vote,
      comment: next.comment,
    }),
  });
  if (!response.ok) throw new Error("HTTP " + response.status);
  const payload = await response.json();
  if (payload && payload.items) state.feedback = payload.items;
}

async function setVote(item, vote) {
  const key = feedbackKey(item);
  const prev = state.feedback[key] ? { ...state.feedback[key] } : { vote: "", comment: "" };
  const nextVote = prev.vote === vote ? "" : vote;
  writeFeedbackLocal(item, { vote: nextVote });
  render();
  try {
    await saveFeedback(item);
  } catch (error) {
    if (!prev.vote && !prev.comment) delete state.feedback[key];
    else state.feedback[key] = prev;
    render();
    showError(error);
  }
}

function openItem() {
  const items = filtered();
  if (state.open < 0 || state.open >= items.length) return null;
  return items[state.open];
}

function step(delta) {
  const items = filtered();
  const next = state.open + delta;
  if (state.open < 0 || next < 0 || next >= items.length) return;
  state.open = next;
  state.page = Math.floor(next / PAGE_SIZE);
  render();
}

async function load(requestedRun) {
  $("summary").textContent = "Loading…";
  const params = new URLSearchParams();
  const run = requestedRun !== undefined ? requestedRun : state.run;
  if (run) params.set("run", run);
  const catalogUrl = "/api/catalog" + (params.toString() ? "?" + params : "");
  const [catalogRes, feedbackRes] = await Promise.all([
    fetch(catalogUrl, { cache: "no-store" }),
    fetch("/api/feedback", { cache: "no-store" }),
  ]);
  if (!catalogRes.ok) throw new Error("HTTP " + catalogRes.status);
  state.catalog = await catalogRes.json();
  if (feedbackRes.ok) {
    const payload = await feedbackRes.json();
    state.feedback = payload.items || {};
  } else {
    state.feedback = {};
  }
  state.run = state.catalog.run || "";
  if (!state.model || !(state.catalog.models || []).some((item) => item.id === state.model)) {
    state.model = state.catalog.models && state.catalog.models[0] ? state.catalog.models[0].id : "";
  }
  if (state.phase && !(state.catalog.phases || []).includes(state.phase)) {
    state.phase = (state.catalog.phases && state.catalog.phases[0]) || "VIBE";
  }
  const phaseOps = (state.catalog.operations || []).filter((item) => item.phase === state.phase);
  if (state.operation && !phaseOps.some((item) => item.id === state.operation)) {
    state.operation = "";
    state.caseKey = "";
  }
  if (
    state.caseKey &&
    !(state.catalog.cases || []).some(
      (item) => item.phase === state.phase && item.operation + "/" + item.id === state.caseKey,
    )
  ) {
    state.caseKey = "";
  }
  $("status").value = state.status;
  state.open = -1;
  renderFilters();
  renderWarnings();
  render();
}

function showError(error) {
  $("summary").textContent = "Could not load: " + (error && error.message ? error.message : error);
}

function initChrome() {
  const reload = $("reload");
  reload.replaceChildren(lucide("refresh-cw", 14), document.createTextNode(" Reload"));
  setIcon($("viewer-like"), "thumbs-up", 16);
  setIcon($("viewer-dislike"), "thumbs-down", 16);
  setIcon($("viewer-comment-btn"), "message-square", 16);
}

$("phase").addEventListener("change", (event) => {
  state.phase = event.target.value;
  state.operation = "";
  state.caseKey = "";
  state.page = 0;
  state.open = -1;
  renderFilters();
  render();
});

$("model").addEventListener("change", (event) => {
  state.model = event.target.value;
  state.page = 0;
  state.open = -1;
  renderFilters();
  render();
});

$("run").addEventListener("change", (event) => {
  const run = event.target.value;
  state.run = run;
  load(run).catch(showError);
});

$("operation").addEventListener("change", (event) => {
  state.operation = event.target.value;
  if (!state.operation) state.caseKey = "";
  else if (state.caseKey && state.caseKey.split("/")[0] !== state.operation) state.caseKey = "";
  state.page = 0;
  state.open = -1;
  renderFilters();
  render();
});

$("case").addEventListener("change", (event) => {
  state.caseKey = event.target.value;
  if (state.caseKey) state.operation = state.caseKey.split("/")[0];
  state.page = 0;
  state.open = -1;
  renderFilters();
  render();
});

$("status").addEventListener("change", (event) => {
  state.status = event.target.value;
  state.page = 0;
  state.open = -1;
  render();
});

$("search").addEventListener("input", (event) => {
  state.query = event.target.value;
  state.page = 0;
  state.open = -1;
  render();
});

$("reload").addEventListener("click", () => {
  load(state.run).catch(showError);
});

function closeViewer() {
  state.open = -1;
  render();
}

$("viewer").addEventListener("click", (event) => {
  if (event.target === $("viewer")) closeViewer();
});
$("viewer-close").addEventListener("click", closeViewer);
$("viewer-prev").addEventListener("click", () => step(-1));
$("viewer-next").addEventListener("click", () => step(1));
$("viewer-gt").addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  const item = openItem();
  if (item) toggleGt(item.key);
});
$("viewer-like").addEventListener("click", () => {
  const item = openItem();
  if (item) setVote(item, "like");
});
$("viewer-dislike").addEventListener("click", () => {
  const item = openItem();
  if (item) setVote(item, "dislike");
});
$("viewer-comment-btn").addEventListener("click", () => {
  const item = openItem();
  if (!item) return;
  const key = feedbackKey(item);
  state.commentOpen = state.commentOpen === key ? "" : key;
  render();
});
$("viewer-comment").addEventListener("input", (event) => {
  const item = openItem();
  if (!item) return;
  writeFeedbackLocal(item, { comment: event.target.value });
});
$("viewer-comment").addEventListener("blur", () => {
  const item = openItem();
  if (item) saveFeedback(item).catch(showError);
});
$("viewer-comment").addEventListener("keydown", (event) => {
  event.stopPropagation();
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    event.target.blur();
  }
});

document.addEventListener("keydown", (event) => {
  if (state.open < 0) return;
  if (event.key === "Escape") {
    closeViewer();
    return;
  }
  const target = event.target;
  if (target && (target.id === "search" || target.tagName === "SELECT" || target.tagName === "TEXTAREA")) return;
  if (event.key === "ArrowLeft") step(-1);
  if (event.key === "ArrowRight") step(1);
});

initChrome();
load().catch(showError);
