"""Index VIBE / Prior samples and batch result images. No HTTP."""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_OP_ORDER = ["add", "remove", "replace", "change", "relocate"]
_PHASES = ("VIBE", "Prior")
_EXPECTED_MODELS = (
    ("flux2-klein-distill", "distill"),
    ("flux2-klein-base", "base"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_catalog(root: Path | None = None, run: str | None = None) -> dict:
    root = (root or repo_root()).resolve()
    warnings: list[str] = []
    models, runs = _discover_models_and_runs(root)
    selected = _pick_run(runs, run)
    if run and runs and selected != run:
        warnings.append(f"Run {run!r} not found; using {selected}.")
    if not runs:
        warnings.append("No run in output/batch yet. Samples still list; result frames stay empty until batch writes images.")

    samples, sample_warnings = _load_samples(root)
    warnings.extend(sample_warnings)

    results_cache: dict[Path, dict[str, dict]] = {}
    items: list[dict] = []
    for sample in samples:
        results = {}
        for model in models:
            results[model["id"]] = _model_result(
                root,
                model["id"],
                selected,
                sample,
                results_cache,
            )
        items.append({**sample, "models": results})

    operations, cases = _summaries(items)
    return {
        "runs": runs,
        "run": selected,
        "models": models,
        "phases": list(_PHASES),
        "operations": operations,
        "cases": cases,
        "warnings": warnings,
        "counts": {"total": len(items)},
        "items": items,
    }


def _discover_models_and_runs(root: Path) -> tuple[list[dict], list[str]]:
    batch = root / "output" / "batch"
    found: list[str] = []
    if batch.is_dir():
        found = sorted(
            path.name
            for path in batch.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    models: list[dict] = []
    seen: set[str] = set()
    for model_id, label in _EXPECTED_MODELS:
        models.append({"id": model_id, "label": label})
        seen.add(model_id)
    for model_id in found:
        if model_id in seen:
            continue
        models.append({"id": model_id, "label": model_id})
        seen.add(model_id)

    runs: set[str] = set()
    for model in models:
        model_dir = batch / model["id"]
        if not model_dir.is_dir():
            continue
        for path in model_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                runs.add(path.name)
    return models, sorted(runs, reverse=True)


def _pick_run(runs: list[str], requested: str | None) -> str:
    if requested and requested in runs:
        return requested
    return runs[0] if runs else ""


def _load_samples(root: Path) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    warnings: list[str] = []
    vibe, vibe_warnings = _load_vibe(root)
    prior, prior_warnings = _load_prior(root)
    warnings.extend(vibe_warnings)
    warnings.extend(prior_warnings)
    items.extend(vibe)
    items.extend(prior)
    return items, warnings


def _load_vibe(root: Path) -> tuple[list[dict], list[str]]:
    data = root / "data" / "VIBE_test"
    warnings: list[str] = []
    items: list[dict] = []
    if not data.is_dir():
        warnings.append("data/VIBE_test not found")
        return items, warnings

    for op_dir in _dirs(data):
        meta_path = op_dir / "metadata.json"
        if not meta_path.is_file():
            warnings.append(f"VIBE/{op_dir.name}: missing metadata.json")
            continue
        meta, error = _read_object(meta_path)
        if error:
            warnings.append(f"VIBE/{op_dir.name}: {error}")
            continue
        operation = op_dir.name
        label = str(meta.get("task") or operation).strip().upper() or operation.upper()
        samples = meta.get("samples")
        if not isinstance(samples, list):
            warnings.append(f"VIBE/{operation}: samples is not a list")
            samples = []
        if isinstance(meta.get("count"), int) and meta["count"] != len(samples):
            warnings.append(f"VIBE/{operation}: count={meta['count']} but {len(samples)} samples")
        for index, sample in enumerate(samples):
            item, extra = _vibe_item(root, op_dir, operation, label, index, sample)
            warnings.extend(extra)
            if item is not None:
                items.append(item)
    return items, warnings


def _vibe_item(
    root: Path,
    op_dir: Path,
    operation: str,
    label: str,
    index: int,
    sample: object,
) -> tuple[dict | None, list[str]]:
    warnings: list[str] = []
    where = f"VIBE/{operation}"
    if not isinstance(sample, dict):
        warnings.append(f"{where}: sample {index + 1} is not an object")
        return None, warnings
    sample_id = str(sample.get("id") or "").strip() or f"sample-{index + 1}"
    prompt = str(sample.get("prompt") or "").strip()
    source_id = str(sample.get("source_id") or "").strip()
    input_public, input_v, input_error, input_name = _resolve_data_image(
        root, op_dir, sample.get("image"), "data/VIBE_test/"
    )
    if input_error:
        warnings.append(f"{where} {sample_id}: input {input_error} ({input_name or 'no path'})")
    output_name = f"{sample_id}.png"
    return (
        _sample_payload(
            phase="VIBE",
            operation=operation,
            operation_label=label,
            case="",
            case_label="",
            case_description="",
            edit_type="",
            sample_id=sample_id,
            result_id=sample_id,
            output_name=output_name,
            prompt=prompt,
            source_id=source_id,
            input_public=input_public,
            input_v=input_v,
            input_name=input_name,
            input_error=input_error,
            gt_public=None,
            gt_v=None,
            gt_name="",
            gt_error="",
        ),
        warnings,
    )


def _load_prior(root: Path) -> tuple[list[dict], list[str]]:
    data = root / "data" / "Prior_test"
    warnings: list[str] = []
    items: list[dict] = []
    if not data.is_dir():
        warnings.append("data/Prior_test not found")
        return items, warnings

    for op_dir in _dirs(data):
        if op_dir.name.lower() == "relocate":
            continue
        for case_dir in _dirs(op_dir):
            case_items, extra = _load_prior_case(root, op_dir, case_dir)
            warnings.extend(extra)
            items.extend(case_items)
    return items, warnings


def _load_prior_case(root: Path, op_dir: Path, case_dir: Path) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    op_id = op_dir.name
    case_file = case_dir.name
    where = f"Prior/{op_id}/{case_file}"
    meta_path = case_dir / "metadata.json"
    if not meta_path.is_file():
        warnings.append(f"{where}: missing metadata.json")
        return [], warnings
    meta, error = _read_object(meta_path)
    if error:
        warnings.append(f"{where}: {error}")
        return [], warnings
    declared = str(meta.get("operation") or "").strip()
    if declared.lower() == "relocate":
        return [], warnings
    op_label = declared or op_id.upper()
    case_label = str(meta.get("case") or case_file).strip() or case_file
    description = str(meta.get("case_description") or "").strip()
    edit_type = str(meta.get("edit_type") or "").strip()
    samples = meta.get("samples")
    if not isinstance(samples, list):
        warnings.append(f"{where}: samples is not a list")
        samples = []
    if isinstance(meta.get("count"), int) and meta["count"] != len(samples):
        warnings.append(f"{where}: count={meta['count']} but {len(samples)} samples")

    items: list[dict] = []
    seen: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        item, extra = _prior_item(
            root,
            case_dir,
            where,
            op_id,
            op_label,
            case_file,
            case_label,
            description,
            edit_type,
            index,
            sample,
            seen,
        )
        warnings.extend(extra)
        if item is not None:
            items.append(item)
    return items, warnings


def _prior_item(
    root: Path,
    case_dir: Path,
    where: str,
    op_id: str,
    op_label: str,
    case_file: str,
    case_label: str,
    description: str,
    edit_type: str,
    index: int,
    sample: object,
    seen: set[str],
) -> tuple[dict | None, list[str]]:
    warnings: list[str] = []
    if not isinstance(sample, dict):
        warnings.append(f"{where}: sample {index} is not an object")
        return None, warnings

    prompt = str(sample.get("text") or "").strip()
    image_rel = sample.get("image")
    file_stem = Path(str(image_rel)).stem if isinstance(image_rel, str) else ""
    if file_stem.startswith("input_"):
        result_id = "output_" + file_stem[len("input_") :]
    else:
        result_id = file_stem or f"sample-{index}"
    if result_id in seen:
        result_id = f"{result_id}_{index}"
    seen.add(result_id)
    display_id = _display_id(result_id, index)
    output_name = f"{result_id}.png"

    input_public, input_v, input_error, input_name = _resolve_data_image(
        root, case_dir, image_rel, "data/Prior_test/"
    )
    gt_public, gt_v, gt_error, gt_name = _resolve_data_image(
        root, case_dir, sample.get("ground_truth"), "data/Prior_test/"
    )
    if input_error:
        warnings.append(f"{where} {display_id}: input {input_error} ({input_name or 'no path'})")
    if gt_error:
        warnings.append(f"{where} {display_id}: ground truth {gt_error} ({gt_name or 'no path'})")

    return (
        _sample_payload(
            phase="Prior",
            operation=op_id,
            operation_label=op_label,
            case=case_file,
            case_label=case_label,
            case_description=description,
            edit_type=edit_type,
            sample_id=display_id,
            result_id=result_id,
            output_name=output_name,
            prompt=prompt,
            source_id="",
            input_public=input_public,
            input_v=input_v,
            input_name=input_name,
            input_error=input_error,
            gt_public=gt_public,
            gt_v=gt_v,
            gt_name=gt_name,
            gt_error=gt_error,
        ),
        warnings,
    )


def _sample_payload(
    *,
    phase: str,
    operation: str,
    operation_label: str,
    case: str,
    case_label: str,
    case_description: str,
    edit_type: str,
    sample_id: str,
    result_id: str,
    output_name: str,
    prompt: str,
    source_id: str,
    input_public: str | None,
    input_v: int | None,
    input_name: str,
    input_error: str,
    gt_public: str | None,
    gt_v: int | None,
    gt_name: str,
    gt_error: str,
) -> dict:
    return {
        "key": f"{phase}/{operation}/{case}/{sample_id}",
        "phase": phase,
        "operation": operation,
        "operation_label": operation_label,
        "case": case,
        "case_label": case_label,
        "case_description": case_description,
        "edit_type": edit_type,
        "id": sample_id,
        "result_id": result_id,
        "output_name": output_name,
        "prompt": prompt,
        "source_id": source_id,
        "input": input_public,
        "input_v": input_v,
        "input_name": input_name,
        "input_error": input_error,
        "gt": gt_public,
        "gt_v": gt_v,
        "gt_name": gt_name,
        "gt_error": gt_error,
        "has_gt": bool(gt_public),
    }


def _model_result(
    root: Path,
    model_id: str,
    run: str,
    sample: dict,
    cache: dict[Path, dict[str, dict]],
) -> dict:
    output_name = sample["output_name"]
    empty = {
        "output": None,
        "output_v": None,
        "output_name": f"results/{output_name}",
        "status": "missing",
        "error": "",
    }
    if not run:
        return empty

    group = _group_dir(root, model_id, run, sample["phase"], sample["operation"], sample["case"])
    if group not in cache:
        cache[group] = _results_index(group)
    record = cache[group].get(sample["result_id"], {})
    json_status = str(record.get("status") or "")
    json_error = str(record.get("error") or "") if record.get("error") else ""

    png = _existing_png(root, group, record.get("output"), output_name)
    if json_status == "error" and png is None:
        return {
            "output": None,
            "output_v": None,
            "output_name": f"results/{output_name}",
            "status": "error",
            "error": json_error,
        }
    if png is None:
        return empty
    public, mtime_ns = png
    return {
        "output": public,
        "output_v": mtime_ns,
        "output_name": f"results/{output_name}",
        "status": "ok",
        "error": "",
    }


def _group_dir(root: Path, model_id: str, run: str, phase: str, operation: str, case: str) -> Path:
    folder = root / "output" / "batch" / model_id / run / phase / operation
    if case:
        folder = folder / case
    return folder


def _results_index(group_dir: Path) -> dict[str, dict]:
    path = group_dir / "results.json"
    if not path.is_file():
        return {}
    data, error = _read_object(path)
    if error:
        return {}
    rows = data.get("results")
    if not isinstance(rows, list):
        return {}
    index: dict[str, dict] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("id"):
            index[str(row["id"])] = row
    return index


def _existing_png(
    root: Path,
    group: Path,
    recorded: object,
    output_name: str,
) -> tuple[str, int] | None:
    candidates: list[Path] = [group / "results" / output_name]
    if isinstance(recorded, str) and recorded.strip():
        rel = recorded.strip().replace("\\", "/")
        parts = tuple(part for part in rel.split("/") if part not in {"", "."})
        if parts and ".." not in parts and not Path(rel).is_absolute():
            candidates.insert(0, group.joinpath(*parts))
    for path in candidates:
        resolved = path.resolve()
        try:
            public = resolved.relative_to(root).as_posix()
        except ValueError:
            continue
        if not public.startswith("output/batch/"):
            continue
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if resolved.is_file() and resolved.stat().st_size > 0:
            return public, resolved.stat().st_mtime_ns
    return None


def _resolve_data_image(
    root: Path,
    base_dir: Path,
    rel: object,
    allowed_prefix: str,
) -> tuple[str | None, int | None, str, str]:
    if not isinstance(rel, str) or not rel.strip():
        return None, None, "missing path", ""
    rel_text = rel.strip().replace("\\", "/")
    parts = tuple(part for part in rel_text.split("/") if part not in {"", "."})
    name = "/".join(parts)
    if Path(rel_text).is_absolute() or ".." in parts or not parts:
        return None, None, "invalid path", name
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None, None, "not an image", name
    path = (base_dir / name).resolve()
    try:
        path.relative_to(base_dir.resolve())
        public = path.relative_to(root).as_posix()
    except ValueError:
        return None, None, "outside sample folder", name
    if not public.startswith(allowed_prefix):
        return None, None, "outside dataset", name
    if not path.is_file():
        return None, None, "missing file", name
    return public, path.stat().st_mtime_ns, "", name


def _summaries(items: list[dict]) -> tuple[list[dict], list[dict]]:
    operations: list[dict] = []
    cases: list[dict] = []
    op_seen: dict[tuple[str, str], dict] = {}
    case_seen: dict[tuple[str, str, str], dict] = {}
    for item in items:
        op_key = (item["phase"], item["operation"])
        op = op_seen.get(op_key)
        if op is None:
            op = {
                "phase": item["phase"],
                "id": item["operation"],
                "label": item["operation_label"],
                "edit_type": item["edit_type"],
                "count": 0,
            }
            op_seen[op_key] = op
            operations.append(op)
        op["count"] += 1
        if item["edit_type"] and not op["edit_type"]:
            op["edit_type"] = item["edit_type"]
        if not item["case"]:
            continue
        case_key = (item["phase"], item["operation"], item["case"])
        case = case_seen.get(case_key)
        if case is None:
            case = {
                "phase": item["phase"],
                "operation": item["operation"],
                "operation_label": item["operation_label"],
                "id": item["case"],
                "label": item["case_label"],
                "description": item["case_description"],
                "edit_type": item["edit_type"],
                "count": 0,
            }
            case_seen[case_key] = case
            cases.append(case)
        case["count"] += 1
    operations.sort(key=lambda row: (_PHASES.index(row["phase"]) if row["phase"] in _PHASES else 9, _op_rank(row["id"])))
    cases.sort(
        key=lambda row: (
            _PHASES.index(row["phase"]) if row["phase"] in _PHASES else 9,
            _op_rank(row["operation"]),
            row["id"],
        )
    )
    return operations, cases


def _display_id(result_id: str, index: int) -> str:
    stem = result_id
    for prefix in ("input_", "output_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem or f"sample-{index}"


def _dirs(root: Path) -> list[Path]:
    paths = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    paths.sort(key=lambda path: (_op_rank(path.name), path.name))
    return paths


def _op_rank(name: str) -> tuple[int, str]:
    try:
        return (_OP_ORDER.index(name), name)
    except ValueError:
        return (len(_OP_ORDER), name)


def _read_object(path: Path) -> tuple[dict, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not read {path.name} ({exc})"
    if not isinstance(data, dict):
        return {}, f"{path.name} is not an object"
    return data, ""
