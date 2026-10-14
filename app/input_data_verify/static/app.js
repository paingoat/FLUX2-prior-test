const PAGE_SIZE = 6;

const state = {
  catalog: null,
  operation: "",
  caseKey: "",
  status: "",
  query: "",
  page: 0,
  open: -1,
};

const $ = (id) => document.getElementById(id);

function caseKeyOf(item) {
  return item.operation + "/" + item.case;
}

function filtered() {
  const items = state.catalog ? state.catalog.items : [];
  const query = state.query.trim().toLowerCase();
  return items.filter((item) => {
    if (state.operation && item.operation !== state.operation) return false;
    if (state.caseKey && caseKeyOf(item) !== state.caseKey) return false;
    if (state.status && item.status !== state.status) return false;
    if (!query) return true;
    return [item.id, item.prompt, item.case_label, item.operation_label, item.edit_type]
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

function chip(label, on, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = on ? "chip on" : "chip";
  button.textContent = label;
  button.setAttribute("aria-pressed", on ? "true" : "false");
  button.addEventListener("click", onClick);
  return button;
}

function selectOperation(id) {
  state.operation = id;
  if (!id) {
    state.caseKey = "";
  } else if (state.caseKey && state.caseKey.split("/")[0] !== id) {
    state.caseKey = "";
  }
  state.page = 0;
  state.open = -1;
  renderFilters();
  render();
}

function selectCase(key) {
  state.caseKey = key;
  if (key) state.operation = key.split("/")[0];
  state.page = 0;
  state.open = -1;
  renderFilters();
  render();
}

function renderFilters() {
  const catalog = state.catalog;
  const ops = $("ops");
  const cases = $("cases");
  if (!catalog) {
    ops.replaceChildren();
    cases.replaceChildren();
    return;
  }

  const opNodes = [
    chip("Tất cả", state.operation === "", () => selectOperation("")),
  ];
  catalog.operations.forEach((operation) => {
    const missing = operation.missing ? " · " + operation.missing + " thiếu" : "";
    opNodes.push(
      chip(
        operation.label + " (" + operation.count + ")" + missing,
        state.operation === operation.id,
        () => selectOperation(operation.id),
      ),
    );
  });
  ops.replaceChildren(...opNodes);

  const selected = catalog.operations.find((item) => item.id === state.operation);
  $("case-label").textContent = selected ? selected.label + " › Case" : "Case";
  if (!selected) {
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = "Chọn một thao tác";
    cases.replaceChildren(hint);
    return;
  }

  const caseNodes = [chip("Tất cả", state.caseKey === "", () => selectCase(""))];
  catalog.cases
    .filter((item) => item.operation === state.operation)
    .forEach((item) => {
      const key = item.operation + "/" + item.id;
      const missing = item.missing ? " · " + item.missing + " thiếu" : "";
      caseNodes.push(chip(item.label + " (" + item.count + ")" + missing, state.caseKey === key, () => selectCase(key)));
    });
  cases.replaceChildren(...caseNodes);
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
  summary.textContent = warnings.length + " cảnh báo";
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
  if (!state.catalog) return "Đang tải…";
  const total = state.catalog.counts.total;
  const missing = items.filter((item) => item.status === "missing").length;
  const shown = items.length === total ? items.length + " mẫu" : items.length + " / " + total + " mẫu";
  return missing ? shown + " · " + missing + " thiếu file" : shown;
}

function renderContext() {
  const line = $("context");
  if (!state.catalog || !state.caseKey) {
    line.hidden = true;
    line.textContent = "";
    return;
  }
  const found = state.catalog.cases.find((item) => item.operation + "/" + item.id === state.caseKey);
  const text = found ? found.description : "";
  line.hidden = !text;
  line.textContent = text;
}

function missingCopy(name, error) {
  const file = name || "không có đường dẫn";
  if (error === "thiếu file") return "Chưa có file\n" + file;
  return (error || "Không có ảnh") + "\n" + file;
}

function pane(label, path, version, name, error) {
  const wrap = document.createElement("div");
  wrap.className = "pane";
  const caption = document.createElement("p");
  caption.className = "pane-label";
  caption.textContent = label;
  const frame = document.createElement("div");
  frame.className = "frame";
  if (path) {
    const image = document.createElement("img");
    image.alt = label;
    image.loading = "lazy";
    image.src = imageUrl("thumb", path, version, 640);
    frame.append(image);
  } else {
    const note = document.createElement("p");
    note.className = "missing-file";
    note.textContent = missingCopy(name, error);
    frame.append(note);
  }
  wrap.append(caption, frame);
  return wrap;
}

function fillFrame(frame, label, path, version, name, error) {
  frame.replaceChildren();
  if (path) {
    const image = document.createElement("img");
    image.alt = label;
    image.src = imageUrl("media", path, version);
    frame.append(image);
    return;
  }
  const note = document.createElement("p");
  note.className = "missing-file";
  note.textContent = missingCopy(name, error);
  frame.append(note);
}

function card(item, index) {
  const article = document.createElement("article");
  article.className = "card" + (item.status === "missing" ? " missing" : "");
  article.tabIndex = 0;

  const top = document.createElement("div");
  top.className = "card-top";
  const kicker = document.createElement("p");
  kicker.className = "kicker";
  kicker.textContent = item.operation_label + " · " + item.case_label + " · " + item.id;
  const badge = document.createElement("span");
  badge.className = "badge " + item.status;
  badge.textContent = item.status === "missing" ? "Thiếu file" : "Đủ ảnh";
  top.append(kicker, badge);

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = item.prompt || "(không có prompt)";
  prompt.title = item.prompt || "";

  const pair = document.createElement("div");
  pair.className = "pair";
  pair.append(
    pane("Input", item.input, item.input_v, item.input_name, item.input_error),
    pane("Ground truth", item.output, item.output_v, item.output_name, item.output_error),
  );

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
  prev.textContent = "Trước";
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
  next.textContent = "Sau";
  next.disabled = state.page >= pages - 1;
  next.addEventListener("click", () => {
    state.page += 1;
    state.open = -1;
    render();
  });
  nav.hidden = false;
  nav.replaceChildren(prev, label, next);
}

function renderViewer(items) {
  const modal = $("viewer");
  const open = state.open >= 0 && state.open < items.length;
  modal.hidden = !open;
  document.body.classList.toggle("modal-open", open);
  if (!open) return;

  const item = items[state.open];
  $("viewer-kicker").textContent = item.operation_label + " · " + item.case_label;
  $("viewer-title").textContent = item.id;
  $("viewer-prompt").textContent = item.prompt || "(không có prompt)";
  const error = $("viewer-error");
  error.hidden = !item.error;
  error.textContent = item.error || "";
  fillFrame($("viewer-input"), "Input " + item.id, item.input, item.input_v, item.input_name, item.input_error);
  fillFrame(
    $("viewer-output"),
    "Ground truth " + item.id,
    item.output,
    item.output_v,
    item.output_name,
    item.output_error,
  );
  $("viewer-pos").textContent = state.open + 1 + " / " + items.length + " · mũi tên để chuyển";
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
    empty.textContent = state.catalog ? "Không có mẫu nào khớp bộ lọc." : "Đang tải…";
    $("grid").replaceChildren(empty);
  } else {
    $("grid").replaceChildren(...slice.map((item, index) => card(item, state.page * PAGE_SIZE + index)));
  }
  renderPager(items.length, pages);
  renderViewer(items);
}

function step(delta) {
  const items = filtered();
  const next = state.open + delta;
  if (state.open < 0 || next < 0 || next >= items.length) return;
  state.open = next;
  state.page = Math.floor(next / PAGE_SIZE);
  render();
}

async function load() {
  $("summary").textContent = "Đang tải…";
  const response = await fetch("/api/catalog", { cache: "no-store" });
  if (!response.ok) throw new Error("HTTP " + response.status);
  state.catalog = await response.json();
  if (state.operation && !state.catalog.operations.some((item) => item.id === state.operation)) {
    state.operation = "";
  }
  if (state.caseKey && !state.catalog.cases.some((item) => item.operation + "/" + item.id === state.caseKey)) {
    state.caseKey = "";
  }
  state.open = -1;
  renderFilters();
  renderWarnings();
  render();
}

function showError(error) {
  $("summary").textContent = "Không tải được catalog: " + (error && error.message ? error.message : error);
}

document.querySelectorAll("[data-status]").forEach((button) => {
  button.addEventListener("click", () => {
    state.status = button.dataset.status || "";
    document.querySelectorAll("[data-status]").forEach((item) => {
      const on = (item.dataset.status || "") === state.status;
      item.classList.toggle("on", on);
      item.setAttribute("aria-pressed", on ? "true" : "false");
    });
    state.page = 0;
    state.open = -1;
    render();
  });
});

$("search").addEventListener("input", (event) => {
  state.query = event.target.value;
  state.page = 0;
  state.open = -1;
  render();
});

$("reload").addEventListener("click", () => {
  load().catch(showError);
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

document.addEventListener("keydown", (event) => {
  if (state.open < 0) return;
  if (event.key === "Escape") {
    closeViewer();
    return;
  }
  if (event.target && event.target.id === "search") return;
  if (event.key === "ArrowLeft") step(-1);
  if (event.key === "ArrowRight") step(1);
});

load().catch(showError);
