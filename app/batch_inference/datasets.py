"""Read VIBE_test and Prior_test metadata into a flat task list.

VIBE images already contain the red scribble, so they are sent as-is.
Prior images are the plain inputs. The prompt is the sample ``text``.
A Prior ``relocate`` operation is skipped; this dataset is not expected to
have one, and the batch runner does not relocate anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    phase: str
    operation: str
    case: str | None
    sample_id: str
    prompt: str
    input_path: Path
    ground_truth: Path | None
    output_name: str
    source_metadata: Path


def load_tasks(data_root: Path, phases: list[str], limit: int | None) -> list[Task]:
    tasks: list[Task] = []
    for phase in phases:
        if phase == "VIBE":
            tasks.extend(load_vibe(data_root / "VIBE_test"))
        elif phase == "Prior":
            tasks.extend(load_prior(data_root / "Prior_test"))
        else:
            raise ValueError(f"Unknown phase {phase!r}.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1.")
        tasks = tasks[:limit]
    return tasks


def load_vibe(root: Path) -> list[Task]:
    if not root.is_dir():
        raise FileNotFoundError(f"VIBE_test not found: {root}")
    tasks: list[Task] = []
    for op_dir in _dirs(root):
        meta_path = op_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = _read_json(meta_path)
        operation = str(meta.get("task") or op_dir.name)
        for sample in meta.get("samples") or []:
            sample_id = str(sample["id"])
            tasks.append(
                Task(
                    phase="VIBE",
                    operation=operation,
                    case=None,
                    sample_id=sample_id,
                    prompt=str(sample["prompt"]),
                    input_path=(op_dir / sample["image"]).resolve(),
                    ground_truth=None,
                    output_name=f"{sample_id}.png",
                    source_metadata=meta_path.resolve(),
                )
            )
    return tasks


def load_prior(root: Path) -> list[Task]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"Prior_test not found: {root}. Run python script/download_prior_test_data.py first."
        )
    tasks: list[Task] = []
    for op_dir in _dirs(root):
        if op_dir.name.lower() == "relocate":
            continue
        for case_dir in _dirs(op_dir):
            meta_path = case_dir / "metadata.json"
            if not meta_path.is_file():
                continue
            meta = _read_json(meta_path)
            if str(meta.get("operation", "")).lower() == "relocate":
                continue
            case = str(meta.get("case_file") or case_dir.name)
            seen: set[str] = set()
            for index, sample in enumerate(meta.get("samples") or [], start=1):
                image_rel = str(sample["image"])
                file_stem = Path(image_rel).stem
                if file_stem.startswith("input_"):
                    sample_id = "output_" + file_stem[len("input_") :]
                else:
                    sample_id = file_stem
                if sample_id in seen:
                    sample_id = f"{sample_id}_{index}"
                seen.add(sample_id)
                gt_rel = sample.get("ground_truth")
                tasks.append(
                    Task(
                        phase="Prior",
                        operation=op_dir.name,
                        case=case,
                        sample_id=sample_id,
                        prompt=str(sample["text"]),
                        input_path=(case_dir / image_rel).resolve(),
                        ground_truth=(case_dir / gt_rel).resolve() if gt_rel else None,
                        output_name=f"{sample_id}.png",
                        source_metadata=meta_path.resolve(),
                    )
                )
    return tasks


def _dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data
