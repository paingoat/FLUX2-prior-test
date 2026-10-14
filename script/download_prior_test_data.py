"""Download Prior_test input/ground-truth images and write per-case metadata.

Source: resource/flux2_klein_60_cases.json
Layout:  data/Prior_test/<operation>/<case_file>/{images,ground_truth,metadata.json}

Each pair keeps the output_image stem, with a role prefix:
input_246644.png in images/, output_246644.png in ground_truth/.
A dataset retry suffix is dropped, so kf05yx_retry1.png is saved as kf05yx.png.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "resource" / "flux2_klein_60_cases.json"
DEFAULT_OUT = ROOT / "data" / "Prior_test"

MAX_ATTEMPTS = 3
TIMEOUT_SEC = 60
RETRY_BACKOFF_SEC = 1.5
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


_RETRY_SUFFIX = re.compile(r"_retry\d+(?=\.[^.]+$)")


def source_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name or name in {".", "/"}:
        raise ValueError(f"Khong lay duoc ten file tu URL: {url}")
    return _RETRY_SUFFIX.sub("", name)


def named_file(source_name: str, prefix: str) -> str:
    return f"{prefix}_{source_name}"


def is_downloaded(dest: Path) -> bool:
    return dest.is_file() and dest.stat().st_size > 0


def download(url: str, dest: Path, attempts: int = MAX_ATTEMPTS) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
                data = response.read()
            if not data:
                raise RuntimeError("response rong")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < attempts:
                time.sleep(RETRY_BACKOFF_SEC * attempt)

    raise RuntimeError(str(last_error))


def render_progress(done: int, total: int, label: str) -> None:
    width = 32
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    line = f"\r[{bar}] {done}/{total} {label}"
    sys.stdout.write(line[:140].ljust(140))
    sys.stdout.flush()


def load_cases(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"JSON phai la mot list: {path}")
    return rows


def group_cases(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        operation = str(row["operation"]).lower()
        case_file = str(row["case_file"])
        grouped[(operation, case_file)].append(row)
    return grouped


def write_metadata(case_dir: Path, operation: str, case_file: str, samples: list[dict], kept: list[dict]) -> None:
    first = samples[0]
    payload = {
        "operation": first["operation"],
        "edit_type": first["edit_type"],
        "case": first["case"],
        "case_file": case_file,
        "case_description": first["case_description"],
        "count": len(kept),
        "samples": kept,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Download Prior_test images into data/Prior_test.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Duong dan file cases JSON")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Thu muc Prior_test")
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS, help="So lan thu moi anh")
    parser.add_argument("--force", action="store_true", help="Tai lai ca anh da co")
    args = parser.parse_args()

    rows = load_cases(args.json)
    grouped = group_cases(rows)
    jobs: list[tuple[str, Path]] = []
    plan: list[tuple[str, str, list[dict]]] = []

    for (operation, case_file), samples in grouped.items():
        case_dir = args.out / operation / case_file
        plan.append((operation, case_file, samples))
        for sample in samples:
            source_name = source_filename(sample["output_image"])
            jobs.append((sample["open_image_input_url"], case_dir / "images" / named_file(source_name, "input")))
            jobs.append((sample["output_image"], case_dir / "ground_truth" / named_file(source_name, "output")))

    total = len(jobs)
    present = 0 if args.force else sum(1 for _, dest in jobs if is_downloaded(dest))
    missing = total - present
    print(f"Done: {present}/{total} anh da co tren local, khong tai lai.")
    if missing == 0:
        print("Khong co anh nao can tai.")
    else:
        print(f"Chi tai {missing} anh con thieu:")
        for _, dest in jobs:
            if args.force or not is_downloaded(dest):
                print(f"- {dest.relative_to(args.out)}")

    done = present
    failures: list[dict] = []
    if missing:
        render_progress(done, total, "da co san")

    for operation, case_file, samples in plan:
        case_dir = args.out / operation / case_file
        kept: list[dict] = []

        for sample in samples:
            source_name = source_filename(sample["output_image"])
            input_name = named_file(source_name, "input")
            output_name = named_file(source_name, "output")
            image_rel = f"images/{input_name}"
            gt_rel = f"ground_truth/{output_name}"
            targets = (
                ("images", sample["open_image_input_url"], case_dir / image_rel, input_name),
                ("ground_truth", sample["output_image"], case_dir / gt_rel, output_name),
            )
            sample_ok = True

            for kind, url, dest, saved_name in targets:
                label = f"{operation}/{case_file}/{kind}/{saved_name}"
                if is_downloaded(dest) and not args.force:
                    continue
                try:
                    download(url, dest, attempts=args.attempts)
                except Exception as exc:
                    sample_ok = False
                    failures.append(
                        {
                            "operation": operation,
                            "case_file": case_file,
                            "kind": kind,
                            "file": saved_name,
                            "url": url,
                            "error": str(exc),
                        }
                    )
                done += 1
                render_progress(done, total, label)

            if sample_ok:
                kept.append(
                    {
                        "text": sample["text"],
                        "image": image_rel,
                        "ground_truth": gt_rel,
                        "open_image_input_url": sample["open_image_input_url"],
                        "output_image": sample["output_image"],
                    }
                )

        write_metadata(case_dir, operation, case_file, samples, kept)

    sys.stdout.write("\n")
    ok = total - len(failures)
    print(f"Xong: {ok}/{total} anh, {len(grouped)} case.")
    if not failures:
        print("Khong co anh loi.")
        return 0

    print(f"Anh loi ({len(failures)}):")
    for item in failures:
        print(
            f"- {item['operation']}/{item['case_file']}/{item['kind']}/{item['file']}\n"
            f"  url: {item['url']}\n"
            f"  loi: {item['error']}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
