"""Local viewer for Prior_test input / ground-truth pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from catalog import IMAGE_SUFFIXES, build_catalog, repo_root

APP_DIR = Path(__file__).resolve().parent
ROOT = repo_root()
STATIC = APP_DIR / "static"
CACHE = APP_DIR / ".cache"
DATA_ROOT = ROOT / "data" / "Prior_test"

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

    def _route(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/catalog":
            self._json(build_catalog(ROOT))
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path.startswith("/thumb/"):
            self._media(path[len("/thumb/") :], thumb=True, query=parse_qs(parsed.query))
            return
        if path.startswith("/media/"):
            self._media(path[len("/media/") :], thumb=False, query=parse_qs(parsed.query))
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

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def _safe_image(rel: str) -> Path | None:
    if not rel or "\\" in rel or rel.startswith("/"):
        return None
    parts = [part for part in rel.split("/") if part not in {"", "."}]
    if ".." in parts or len(parts) < 4:
        return None
    if parts[0] != "data" or parts[1] != "Prior_test":
        return None
    if Path(parts[-1]).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    path = (ROOT.joinpath(*parts)).resolve()
    try:
        path.relative_to(DATA_ROOT.resolve())
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path


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
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Xem input và ground truth của Prior_test.")
    parser.add_argument("--host", default=os.environ.get("VIEW_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VIEW_PORT", "8501")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Prior test viewer: {url}")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} user@host")
    print("Tải lại trang hoặc bấm Tải lại để đọc lại metadata.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
