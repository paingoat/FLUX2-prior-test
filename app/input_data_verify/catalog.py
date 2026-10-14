"""Index Prior_test samples from per-case metadata. No HTTP."""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_OP_ORDER = ["add", "remove", "replace", "change", "relocate"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_catalog(root: Path | None = None) -> dict:
    root = (root or repo_root()).resolve()
    data = root / "data" / "Prior_test"
    warnings: list[str] = []
    items: list[dict] = []
    operations: list[dict] = []
    cases: list[dict] = []

    if not data.is_dir():
        warnings.append(f"Không tìm thấy {data.relative_to(root).as_posix()}")
        return _payload(operations, cases, warnings, items)

    op_dirs = [path for path in data.iterdir() if path.is_dir()]
    op_dirs.sort(key=lambda path: (_op_rank(path.name), path.name))

    for op_dir in op_dirs:
        op_id = op_dir.name
        op_items_start = len(items)
        edit_types: list[str] = []
        op_label = op_id.upper()

        case_dirs = sorted((path for path in op_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
        for case_dir in case_dirs:
            case_items, case_info, case_warnings, label = _load_case(root, op_dir, case_dir)
            warnings.extend(case_warnings)
            if case_info is None:
                continue
            op_label = label or op_label
            if case_info["edit_type"] and case_info["edit_type"] not in edit_types:
                edit_types.append(case_info["edit_type"])
            cases.append(case_info)
            items.extend(case_items)

        op_items = items[op_items_start:]
        if not op_items and not any(case["operation"] == op_id for case in cases):
            continue
        operations.append(
            {
                "id": op_id,
                "label": op_label,
                "edit_type": edit_types[0] if len(edit_types) == 1 else "",
                "count": len(op_items),
                "missing": sum(1 for item in op_items if item["status"] == "missing"),
            }
        )

    return _payload(operations, cases, warnings, items)


def _load_case(root: Path, op_dir: Path, case_dir: Path) -> tuple[list[dict], dict | None, list[str], str]:
    warnings: list[str] = []
    op_id = op_dir.name
    case_file = case_dir.name
    where = f"{op_id}/{case_file}"
    meta_path = case_dir / "metadata.json"
    if not meta_path.is_file():
        warnings.append(f"{where}: thiếu metadata.json")
        return [], None, warnings, ""

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{where}: không đọc được metadata.json ({exc})")
        return [], None, warnings, ""
    if not isinstance(meta, dict):
        warnings.append(f"{where}: metadata.json không phải object")
        return [], None, warnings, ""

    declared = str(meta.get("operation") or "").strip()
    op_label = declared or op_id.upper()
    if declared and declared.lower() != op_id:
        warnings.append(f"{where}: operation trong metadata là {declared}")

    case_label = str(meta.get("case") or case_file).strip() or case_file
    declared_file = str(meta.get("case_file") or "").strip()
    if declared_file and declared_file != case_file:
        warnings.append(f"{where}: case_file trong metadata là {declared_file}")

    description = str(meta.get("case_description") or "").strip()
    edit_type = str(meta.get("edit_type") or "").strip()
    samples = meta.get("samples")
    if not isinstance(samples, list):
        warnings.append(f"{where}: samples không phải list")
        samples = []
    if isinstance(meta.get("count"), int) and meta["count"] != len(samples):
        warnings.append(f"{where}: count={meta['count']} nhưng có {len(samples)} mẫu")

    referenced: set[str] = set()
    items: list[dict] = []
    for index, sample in enumerate(samples):
        item, sample_warnings, names = _sample_item(
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
        )
        warnings.extend(sample_warnings)
        referenced.update(names)
        if item is not None:
            items.append(item)

    warnings.extend(_orphan_warnings(case_dir, where, referenced))
    missing = sum(1 for item in items if item["status"] == "missing")
    case_info = {
        "operation": op_id,
        "operation_label": op_label,
        "id": case_file,
        "label": case_label,
        "description": description,
        "edit_type": edit_type,
        "count": len(items),
        "missing": missing,
    }
    return items, case_info, warnings, op_label


def _sample_item(
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
) -> tuple[dict | None, list[str], set[str]]:
    warnings: list[str] = []
    if not isinstance(sample, dict):
        warnings.append(f"{where}: mẫu {index + 1} không phải object")
        return None, warnings, set()

    prompt = str(sample.get("text") or "").strip()
    input_rel = sample.get("image")
    output_rel = sample.get("ground_truth")
    input_public, input_v, input_error, input_name = _resolve_image(root, case_dir, input_rel)
    output_public, output_v, output_error, output_name = _resolve_image(root, case_dir, output_rel)

    names = {name for name in (input_name, output_name) if name}
    problems = []
    if input_error:
        problems.append(f"input: {input_error}")
        warnings.append(f"{where} mẫu {index + 1}: input {input_error} ({input_name or 'không có đường dẫn'})")
    if output_error:
        problems.append(f"ground truth: {output_error}")
        warnings.append(f"{where} mẫu {index + 1}: ground truth {output_error} ({output_name or 'không có đường dẫn'})")

    return (
        {
            "operation": op_id,
            "operation_label": op_label,
            "edit_type": edit_type,
            "case": case_file,
            "case_label": case_label,
            "case_description": description,
            "id": _sample_id(input_name or output_name, index),
            "prompt": prompt,
            "status": "missing" if problems else "complete",
            "error": "; ".join(problems),
            "input": input_public,
            "output": output_public,
            "input_v": input_v,
            "output_v": output_v,
            "input_name": input_name,
            "output_name": output_name,
            "input_error": input_error,
            "output_error": output_error,
        },
        warnings,
        names,
    )


def _resolve_image(root: Path, case_dir: Path, rel: object) -> tuple[str | None, int | None, str, str]:
    if not isinstance(rel, str) or not rel.strip():
        return None, None, "thiếu đường dẫn", ""
    rel_text = rel.strip().replace("\\", "/")
    parts = tuple(part for part in rel_text.split("/") if part not in {"", "."})
    name = "/".join(parts)
    if Path(rel_text).is_absolute() or ".." in parts or not parts:
        return None, None, "đường dẫn không hợp lệ", name
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None, None, "không phải ảnh", name

    path = (case_dir / name).resolve()
    try:
        path.relative_to(case_dir.resolve())
        public = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return None, None, "nằm ngoài case", name
    if not public.startswith("data/Prior_test/"):
        return None, None, "nằm ngoài Prior_test", name
    if not path.is_file():
        return None, None, "thiếu file", name
    return public, path.stat().st_mtime_ns, "", name


def _orphan_warnings(case_dir: Path, where: str, referenced: set[str]) -> list[str]:
    warnings: list[str] = []
    orphans: list[str] = []
    for folder in ("images", "ground_truth"):
        directory = case_dir / folder
        if not directory.is_dir():
            warnings.append(f"{where}: thiếu thư mục {folder}/")
            continue
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = f"{folder}/{path.name}"
            if rel not in referenced:
                orphans.append(rel)
    if orphans:
        shown = ", ".join(orphans[:5])
        extra = f" (+{len(orphans) - 5})" if len(orphans) > 5 else ""
        warnings.append(f"{where}: ảnh không có trong metadata: {shown}{extra}")
    return warnings


def _sample_id(name: str, index: int) -> str:
    stem = Path(name).stem if name else ""
    for prefix in ("input_", "output_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem or f"sample-{index + 1}"


def _op_rank(name: str) -> tuple[int, str]:
    try:
        return (_OP_ORDER.index(name), name)
    except ValueError:
        return (len(_OP_ORDER), name)


def _payload(operations: list[dict], cases: list[dict], warnings: list[str], items: list[dict]) -> dict:
    return {
        "operations": operations,
        "cases": cases,
        "warnings": warnings,
        "counts": {
            "total": len(items),
            "complete": sum(1 for item in items if item["status"] == "complete"),
            "missing": sum(1 for item in items if item["status"] == "missing"),
        },
        "items": items,
    }
