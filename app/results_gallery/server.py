"""Local viewer for batch results on VIBE_test and Prior_test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from catalog import IMAGE_SUFFIXES, build_catalog, repo_root

APP_DIR = Path(__file__).resolve().parent
ROOT = repo_root()
STATIC = APP_DIR / "static"
CACHE = APP_DIR / ".cache"
FEEDBACK_PATH = APP_DIR / "feedback.json"
_FEEDBACK_LOCK = threading.Lock()
_MAX_BODY = 32 * 1024
_MAX_COMMENT = 2000
_KEY_FIELDS = ("run", "model", "phase", "operation", "case", "id")

_PUBLIC_ROOTS = (
    ROOT / "data" / "VIBE_test",
    ROOT / "data" / "Prior_test",
    ROOT / "output" / "batch",
)

STATIC_MAP = {
    "/": STATIC / "index.html",
    "/index.html": STATIC / "index.html",
    "/static/app.js": STATIC / "app.js",
    "/static/style.css": STATIC / "style.css",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/api/feedback":
                self._save_feedback()
                return
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _route(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path == "/api/catalog":
            run = (query.get("run") or [""])[0].strip() or None
            self._json(build_catalog(ROOT, run=run))
            return
        if path == "/api/feedback":
            self._json(_read_feedback())
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path.startswith("/thumb/"):
            self._media(path[len("/thumb/") :], thumb=True, query=query)
            return
        if path.startswith("/media/"):
            self._media(path[len("/media/") :], thumb=False, query=query)
            return
        if path in STATIC_MAP:
            self._static(STATIC_MAP[path])
            return
        self.send_error(404)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        self._send_file(path, CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"), cache="no-cache")

    def _media(self, rel: str, *, thumb: bool, query: dict[str, list[str]]) -> None:
        src = _safe_image(rel)
        if src is None:
            self.send_error(404)
            return
        target = src
        content_type = CONTENT_TYPES.get(src.suffix.lower(), "application/octet-stream")
        if thumb:
            edge = _edge(query.get("e", ["640"])[0])
            try:
                target = _thumbnail(src, edge)
                content_type = "image/jpeg" if target.suffix.lower() in {".jpg", ".jpeg"} else content_type
            except Exception:
                target = src
        self._send_file(target, content_type, cache="public, max-age=86400")

    def _send_file(self, path: Path, content_type: str, *, cache: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _save_feedback(self) -> None:
        raw = _read_body(self)
        if raw is None:
            self._json({"error": "payload too large"}, status=413)
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json({"error": "invalid JSON"}, status=400)
            return
        if not isinstance(payload, dict):
            self._json({"error": "expected an object"}, status=400)
            return
        try:
            stored = _upsert_feedback(payload)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return
        self._json(stored)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def _safe_image(rel: str) -> Path | None:
    if not rel or "\\" in rel or rel.startswith("/"):
        return None
    parts = [part for part in rel.split("/") if part not in {"", "."}]
    if ".." in parts or len(parts) < 4:
        return None
    prefix = tuple(parts[:2])
    if prefix not in {("data", "VIBE_test"), ("data", "Prior_test"), ("output", "batch")}:
        return None
    if Path(parts[-1]).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    path = (ROOT.joinpath(*parts)).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return None
    for allowed in _PUBLIC_ROOTS:
        try:
            path.relative_to(allowed.resolve())
            break
        except ValueError:
            continue
    else:
        return None
    if not path.is_file():
        return None
    return path


def _read_body(handler: Handler) -> bytes | None:
    try:
        length = int(handler.headers.get("Content-Length") or "0")
    except ValueError:
        return b""
    if length < 0:
        return b""
    if length > _MAX_BODY:
        remaining = length
        while remaining > 0:
            chunk = handler.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        return None
    return handler.rfile.read(length) if length else b""


def _feedback_key(fields: dict[str, str]) -> str:
    parts = []
    for name in _KEY_FIELDS:
        value = fields[name]
        if "|" in value:
            raise ValueError(f"invalid {name}")
        parts.append(value)
    if not fields["model"] or not fields["phase"] or not fields["operation"] or not fields["id"]:
        raise ValueError("model, phase, operation, and id are required")
    if fields["phase"] not in {"VIBE", "Prior"}:
        raise ValueError("invalid phase")
    return "|".join(parts)


def _read_feedback() -> dict:
    if not FEEDBACK_PATH.is_file():
        return {"items": {}}
    try:
        data = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"items": {}}
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, dict):
        return {"items": {}}
    clean = {}
    for key, row in items.items():
        if not isinstance(key, str) or not isinstance(row, dict):
            continue
        vote = row.get("vote") if row.get("vote") in {"like", "dislike"} else ""
        comment = str(row.get("comment") or "")[:_MAX_COMMENT]
        if vote or comment:
            clean[key] = {"vote": vote, "comment": comment}
    return {"items": clean}


def _upsert_feedback(payload: dict) -> dict:
    fields = {}
    for name in _KEY_FIELDS:
        value = payload.get(name, "")
        if not isinstance(value, str):
            raise ValueError(f"invalid {name}")
        fields[name] = value.strip()
    key = _feedback_key(fields)
    vote = payload.get("vote", "")
    if vote is None:
        vote = ""
    if vote not in {"", "like", "dislike"}:
        raise ValueError("invalid vote")
    comment = payload.get("comment", "")
    if comment is None:
        comment = ""
    if not isinstance(comment, str):
        raise ValueError("invalid comment")
    comment = comment[:_MAX_COMMENT]
    with _FEEDBACK_LOCK:
        store = _read_feedback()
        items = store["items"]
        if not vote and not comment:
            items.pop(key, None)
        else:
            items[key] = {"vote": vote, "comment": comment}
        _write_json(FEEDBACK_PATH, store)
        return store


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _edge(raw: str) -> int:
    try:
        edge = int(raw)
    except (TypeError, ValueError):
        return 640
    return max(64, min(1280, edge))


def _thumbnail(src: Path, edge: int) -> Path:
    from PIL import Image, ImageOps

    mtime_ns = src.stat().st_mtime_ns
    digest = hashlib.sha1(f"{src.resolve().as_posix()}:{mtime_ns}:{edge}".encode()).hexdigest()
    dest = CACHE / f"{digest}.jpg"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((edge, edge), resample)
        tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.tmp")
        image.save(tmp, "JPEG", quality=75)
        tmp.replace(dest)
    return dest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser(description="Browse VIBE / Prior batch results.")
    parser.add_argument("--host", default=os.environ.get("VIEW_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VIEW_PORT", "8502")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Results gallery: {url}", flush=True)
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} user@host", flush=True)
    print("Reload the page or click Reload after batch writes new images.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
