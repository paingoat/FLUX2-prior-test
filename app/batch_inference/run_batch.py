"""Batch edit over VIBE_test, then Prior_test, for one or both Klein models.

Output layout::

    output/batch/<model>/run-<timestamp>/<VIBE|Prior>/<operation>[/<case>]/
        results/
        results.json

Re-running the same ``--run-name`` skips images that already have an ``ok``
record and a non-empty file, and rewrites ``results.json`` after every image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict, Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.env import load_env

load_env()

from PIL import Image
from tqdm import tqdm

from app.batch_inference.datasets import Task, load_tasks
from app.common.env import rel_posix, timestamp
from app.common.flux2 import MAX_AREA, MODEL_SPECS, ModelManager

PHASE_NAMES = {"vibe": "VIBE", "prior": "Prior"}
_SOURCE_KEYS = (
    "task",
    "operation",
    "edit_type",
    "case",
    "case_file",
    "case_description",
    "count",
    "note",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch FLUX.2 [klein] edits on VIBE_test and Prior_test.")
    parser.add_argument("--models", nargs="+", default=["distill", "base"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--phases", nargs="+", default=["vibe", "prior"], help="vibe, prior, or both.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--run-name", default=None, help="Default: run-<timestamp>, shared by every model.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N images per model (smoke test).")
    parser.add_argument("--offload", action="store_true", help="CPU-offload one pipeline. For GPUs below 24 GB.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "batch")
    args = parser.parse_args()

    phases = _phases(args.phases)
    models = _dedupe(args.models)
    run_name = args.run_name or f"run-{timestamp()}"
    tasks = load_tasks(args.data_root, phases, args.limit)
    if not tasks:
        print("No images to run.")
        return 1

    print(f"Run {run_name}")
    print(f"Models: {', '.join(models)}")
    print(f"Phases: {', '.join(phases)}")
    print(f"Images per model: {len(tasks)}")
    print(f"Output: {args.output}")

    manager = ModelManager(offload=args.offload)
    failures = 0
    for variant in models:
        failures += _run_variant(
            manager=manager,
            variant=variant,
            tasks=tasks,
            output_root=args.output,
            run_name=run_name,
            phases=phases,
            seed=args.seed,
            offload=args.offload,
            limit=args.limit,
        )
        manager.unload(variant)
    return 1 if failures else 0


def _run_variant(
    manager: ModelManager,
    variant: str,
    tasks: list[Task],
    output_root: Path,
    run_name: str,
    phases: list[str],
    seed: int,
    offload: bool,
    limit: int | None,
) -> int:
    spec = MODEL_SPECS[variant]
    label = spec.dir_name
    run_dir = output_root / label / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        _write_json(
            config_path,
            {
                "run_name": run_name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "variant": spec.key,
                "repo": spec.repo,
                "dir_name": spec.dir_name,
                "phases": phases,
                "seed": seed,
                "offload": offload,
                "limit": limit,
                "steps": spec.steps,
                "guidance": spec.guidance,
                "guidance_fixed": spec.fixed,
                "dtype": "bfloat16",
                "max_area": MAX_AREA,
            },
        )

    tqdm.write(f"[{label}] loading {spec.repo}")
    manager.load(variant, quiet=True)
    nested = _nest(tasks)
    stats: Counter[str] = Counter()
    started = time.perf_counter()
    bar = tqdm(total=len(tasks), desc=label, unit="img")
    try:
        for phase, operations in nested.items():
            phase_tasks = [task for cases in operations.values() for group in cases.values() for task in group]
            tqdm.write(f"[{label}] phase {phase} start ({len(phase_tasks)} images)")
            phase_stats: Counter[str] = Counter()
            phase_started = time.perf_counter()
            for operation, cases in operations.items():
                op_tasks = [task for group in cases.values() for task in group]
                tqdm.write(f"[{label}] phase {phase} / {operation} start ({len(op_tasks)} images)")
                op_stats: Counter[str] = Counter()
                op_started = time.perf_counter()
                for case, group in cases.items():
                    case_started = time.perf_counter()
                    if case is not None:
                        tqdm.write(
                            f"[{label}] phase {phase} / {operation} / {case} start ({len(group)} images)"
                        )
                    group_stats = _run_group(manager, variant, group, run_dir, seed, bar)
                    op_stats.update(group_stats)
                    if case is not None:
                        tqdm.write(
                            f"[{label}] phase {phase} / {operation} / {case} done: "
                            f"{_format_stats(group_stats)}, {_seconds(case_started)}"
                        )
                phase_stats.update(op_stats)
                tqdm.write(
                    f"[{label}] phase {phase} / {operation} done: "
                    f"{_format_stats(op_stats)}, {_seconds(op_started)}"
                )
            stats.update(phase_stats)
            tqdm.write(
                f"[{label}] phase {phase} done: {_format_stats(phase_stats)}, {_seconds(phase_started)}"
            )
    finally:
        bar.close()
    tqdm.write(f"[{label}] finished: {_format_stats(stats)}, {_seconds(started)}")
    return stats["error"]


def _run_group(
    manager: ModelManager,
    variant: str,
    tasks: list[Task],
    run_dir: Path,
    seed: int,
    bar: tqdm,
) -> Counter[str]:
    spec = MODEL_SPECS[variant]
    folder = _group_dir(run_dir, tasks[0])
    results_dir = folder / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = folder / "results.json"

    by_id: dict[str, dict] = {}
    order: list[str] = []
    if results_path.is_file():
        try:
            previous = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        for item in previous.get("results") or []:
            item_id = item.get("id")
            if item_id and item_id not in by_id:
                by_id[item_id] = item
                order.append(item_id)

    stats: Counter[str] = Counter()
    for task in tasks:
        dest = results_dir / task.output_name
        prior = by_id.get(task.sample_id)
        bar.set_postfix_str(_postfix(task), refresh=False)
        if dest.is_file() and dest.stat().st_size > 0 and prior and prior.get("status") == "ok":
            stats["skipped"] += 1
            bar.update(1)
            continue

        record = _infer(manager, variant, task, dest, seed)
        if task.sample_id not in by_id:
            order.append(task.sample_id)
        by_id[task.sample_id] = record
        stats[record["status"]] += 1
        _write_json(results_path, _payload(spec, tasks[0], seed, [by_id[key] for key in order]))
        bar.update(1)

    if tasks and not results_path.is_file():
        _write_json(results_path, _payload(spec, tasks[0], seed, [by_id[key] for key in order if key in by_id]))
    return stats


def _infer(manager: ModelManager, variant: str, task: Task, dest: Path, seed: int) -> dict:
    started = time.perf_counter()
    record = {
        "id": task.sample_id,
        "prompt": task.prompt,
        "input": rel_posix(task.input_path),
        "ground_truth": rel_posix(task.ground_truth) if task.ground_truth else None,
        "output": None,
        "width": None,
        "height": None,
        "seed": seed,
        "elapsed_sec": None,
        "status": "error",
        "error": None,
    }
    try:
        if not task.input_path.is_file():
            raise FileNotFoundError(f"missing input image: {task.input_path}")
        with Image.open(task.input_path) as image:
            rgb = image.convert("RGB")
            rgb.load()
        result, meta = manager.edit(
            variant=variant,
            image=rgb,
            prompt=task.prompt,
            seed=seed,
            quiet=True,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.save(dest)
        record.update(
            output=f"results/{task.output_name}",
            width=result.width,
            height=result.height,
            elapsed_sec=meta["elapsed_sec"],
            status="ok",
        )
    except Exception as exc:
        record["elapsed_sec"] = round(time.perf_counter() - started, 3)
        record["error"] = f"{type(exc).__name__}: {exc}"
        tqdm.write(f"ERROR {task.phase}/{task.operation}/{task.case or '-'}/{task.sample_id}: {record['error']}")
        tqdm.write(traceback.format_exc().rstrip())
    return record


def _payload(spec, task: Task, seed: int, results: list[dict]) -> dict:
    return {
        "model": {
            "variant": spec.key,
            "repo": spec.repo,
            "dir_name": spec.dir_name,
        },
        "phase": task.phase,
        "operation": task.operation,
        "case": task.case,
        "source": _source_block(task.source_metadata),
        "config": {
            "steps": spec.steps,
            "guidance": spec.guidance,
            "guidance_fixed": spec.fixed,
            "seed": seed,
            "dtype": "bfloat16",
            "max_area": MAX_AREA,
        },
        "results": results,
    }


def _source_block(path: Path) -> dict:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"metadata": rel_posix(path), "error": str(exc)}
    block = {key: meta[key] for key in _SOURCE_KEYS if isinstance(meta, dict) and key in meta}
    block["metadata"] = rel_posix(path)
    return block


def _group_dir(run_dir: Path, task: Task) -> Path:
    folder = run_dir / task.phase / task.operation
    if task.case is not None:
        folder = folder / task.case
    return folder


def _nest(tasks: list[Task]) -> OrderedDict[str, OrderedDict[str, OrderedDict[str | None, list[Task]]]]:
    phases: OrderedDict[str, OrderedDict[str, OrderedDict[str | None, list[Task]]]] = OrderedDict()
    for task in tasks:
        operations = phases.setdefault(task.phase, OrderedDict())
        cases = operations.setdefault(task.operation, OrderedDict())
        cases.setdefault(task.case, []).append(task)
    return phases


def _postfix(task: Task) -> str:
    case = task.case or "-"
    text = f"{task.phase}/{task.operation}/{case}/{task.sample_id}"
    return text if len(text) <= 80 else text[:77] + "..."


def _format_stats(stats: Counter[str]) -> str:
    return f"{stats['ok']} ok, {stats['error']} error, {stats['skipped']} skipped"


def _seconds(started: float) -> str:
    return f"{time.perf_counter() - started:.1f}s"


def _phases(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key not in PHASE_NAMES:
            known = ", ".join(PHASE_NAMES)
            raise SystemExit(f"Unknown phase {name!r}. Choose from: {known}.")
        label = PHASE_NAMES[key]
        if label not in out:
            out.append(label)
    return out


def _dedupe(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
