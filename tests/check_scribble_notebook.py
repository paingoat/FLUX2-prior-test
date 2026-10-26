"""Offline checks for the Colab notebook; no model downloads or GPU needed.

Run: python tests/check_scribble_notebook.py
Requires the notebook's numpy, Pillow, OpenCV and nbformat dependencies.
"""
import ast
import json
import sys
import tempfile
from pathlib import Path

import cv2
import nbformat
import numpy as np
from PIL import Image, ImageDraw


def main():
    notebook_name = sys.argv[1] if len(sys.argv) > 1 else "prior_pairs_to_scribble_colab.ipynb"
    path = Path(__file__).resolve().parents[1] / "notebooks" / notebook_name
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    codes = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    for source in codes:
        ast.parse(source)
    ns = dict(np=np, cv2=cv2, Image=Image, ImageDraw=ImageDraw,
              MIN_COMPONENT_FRACTION=0.001, BOX_PADDING=0.08, MAX_MASK_FRACTION=0.65)
    functions = next(s for s in codes if s.startswith("def mask_and_boxes"))
    exec(compile(functions, str(path), "exec"), ns)
    heat = np.ones((32, 32), np.float32)
    assert "large_mask" in ns["mask_and_boxes"](heat, 0.5)[2]

    # Execute only data normalization definitions, without importing requests or downloading.
    import hashlib
    from urllib.parse import urljoin, urlparse
    ns.update(hashlib=hashlib, urljoin=urljoin, urlparse=urlparse,
              BASE_URL="https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/")
    source = next(s for s in codes if "def normalize_record" in s)
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
             and node.name in {"normalize_record", "operation_of"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    for text, expected in (("Remove an existing object", "remove"),
                           ("Add a new object to the scene", "add"),
                           ("Replace an object", "replace"),
                           ("Color change", "change"),
                           ("Change overall color tone (warm to cool)", None),
                           ("Add film grain or vintage filter", None),
                           ("Relocate an object", None)):
        assert ns["operation_of"](text) == expected
    raw = dict(open_image_input_url="https://example.org/a.jpg", text="Remove the flag.",
               output_image="images/positive-edit/1.png", edit_type="Remove an existing object",
               summarized_text="Remove flag.")
    rec = ns["normalize_record"](raw)
    assert rec["result_url"] == ns["BASE_URL"] + raw["output_image"]
    assert rec["prompt_short"] == raw["summarized_text"]
    assert ns["normalize_record"]({**raw, "output_image": ""}) is None

    # Export synthetic pairs with a fake heatmap: check actual PNG/JSONL/ZIP and split grouping.
    import csv
    import random
    import zipfile
    from pathlib import PurePosixPath
    import shutil
    import subprocess
    from collections import Counter
    with tempfile.TemporaryDirectory(prefix="scribble-check-", dir=path.parent) as directory:
        root = Path(directory)
        for folder in ("original", "result"):
            (root / folder).mkdir()
        rec["id"] = "1234567890abcdef"
        a = np.zeros((80, 120, 3), np.uint8)
        b = a.copy()
        b[10:30, 5:25] = 255
        for folder, image in (("original", a), ("result", b)):
            Image.fromarray(image).save(root / folder / f"{rec['id']}.png")
        heat = np.zeros((80, 120), np.float32)
        heat[10:30, 5:25] = 1
        # Exercise both accepted ZIP schemas and reject path traversal.
        ns.update(Path=Path, PurePosixPath=PurePosixPath, zipfile=zipfile,
                  shutil=shutil, random=random, Counter=Counter, ImageOps=__import__("PIL.ImageOps", fromlist=["ImageOps"]),
                  json=json, SEED=42, N_PER_OPERATION=2, MAX_IMAGE_SIDE=1024,
                  MAX_ASPECT_RATIO_ERROR=0.10)
        ns["write_json"] = lambda p, obj: Path(p).write_text(json.dumps(obj), encoding="utf-8")
        loader_node = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                       and node.name in {"load_uploaded_pairs", "prepare_images"}]
        exec(compile(ast.Module(body=loader_node, type_ignores=[]), str(path), "exec"), ns)
        custom_zip = root / "custom.zip"
        with zipfile.ZipFile(custom_zip, "w") as z:
            z.write(root / "original" / f"{rec['id']}.png", "original/a.png")
            z.write(root / "result" / f"{rec['id']}.png", "result/a.png")
            z.writestr("pairs.jsonl", json.dumps(dict(operation="remove", prompt_original="Remove flag.",
                prompt_short="Remove flag.", original_file_name="original/a.png", result_file_name="result/a.png")))
        loaded, config = ns["load_uploaded_pairs"](custom_zip, root)
        assert len(loaded) == 1 and loaded[0]["prompt_original"] == "Remove flag."
        assert loaded[0]["source_url"].startswith("upload:sha256:")
        prior_zip = root / "prior.zip"
        with zipfile.ZipFile(prior_zip, "w") as z:
            z.write(root / "original" / f"{rec['id']}.png", "Prior_test/remove/case/images/a.png")
            z.write(root / "result" / f"{rec['id']}.png", "Prior_test/remove/case/ground_truth/b.png")
            z.writestr("Prior_test/remove/case/metadata.json", json.dumps(dict(operation="REMOVE", samples=[
                dict(text="Remove flag.", image="images/a.png", ground_truth="ground_truth/b.png") ])))
        loaded, _ = ns["load_uploaded_pairs"](prior_zip, root)
        assert len(loaded) == 1 and loaded[0]["operation"] == "remove"
        # Small ratio changes are retained and flagged, rather than silently dropping all pairs.
        aa, bb, alignment = ns["prepare_images"](Image.new("RGB", (400, 300)), Image.new("RGB", (1184, 864)))
        assert aa.size == bb.size
        assert "aspect_ratio_resized" in alignment["alignment_flags"]
        try:
            ns["prepare_images"](Image.new("RGB", (400, 300)), Image.new("RGB", (100, 300)))
        except ValueError:
            pass
        else:
            raise AssertionError("Strong aspect ratio mismatch should require review")
        evil_zip = root / "evil.zip"
        with zipfile.ZipFile(evil_zip, "w") as z:
            z.writestr("../escaped.txt", "bad")
        try:
            ns["load_uploaded_pairs"](evil_zip, root)
        except ValueError as exc:
            assert "Unsafe ZIP path" in str(exc)
        else:
            raise AssertionError("ZIP traversal was not rejected")
        ns.update(ROOT=root, EXPORT_METHOD="sigma_visual", SEED=42, N_VARIANTS=3,
                  MASK_THRESHOLD=0.5, PIXEL_THRESHOLD=20/255, SIGMA_REV="test",
                  SCRIBBLE_STYLES=["box", "ellipse", "freehand_loop", "cross", "hatch", "short_strokes"],
                  SCRIBBLE_VERSION="test-v2",
                  checkpoint_revision="test", reviews={}, pairs=[rec], selection_config={},
                  read_pair=lambda _: (a, b), predict_pair=lambda *_: heat,
                  review_key=lambda r, m: m + ":" + r["id"],
                  json=json, csv=csv, shutil=shutil, subprocess=subprocess, sys=sys, Counter=Counter,
                  display=lambda *_: None, FileLink=str)
        ns["write_json"] = lambda p, obj: Path(p).write_text(json.dumps(obj), encoding="utf-8")
        export = next(s for s in codes if s.startswith("APPROVED_ONLY ="))
        exec(compile(export, str(path), "exec"), ns)
        rows = [json.loads(line) for line in (ns["export_dir"] / "metadata.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        assert len({r["split"] for r in rows}) == 1
        assert all(r["review_status"] == "pending" for r in rows)
        assert Path(ns["archive"]).is_file()
        # Rejecting a pair excludes it from export; approval-only excludes pending.
        exec(compile(export.replace("APPROVED_ONLY = False", "APPROVED_ONLY = True"), str(path), "exec"), ns)
        assert ns["rows"] == []
    print("Notebook syntax, schema, masks, scribbles, and export checks passed.")


if __name__ == "__main__":
    main()
