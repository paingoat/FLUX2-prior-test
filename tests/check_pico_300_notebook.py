"""Offline end-to-end checks for the dataset builder; no GPU or downloads.

Run: python tests/check_pico_300_notebook.py
"""
import ast
import csv
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

import cv2
import nbformat
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def main():
    path = Path(__file__).resolve().parents[1] / "notebooks/build_pico_scribble_300_colab.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    codes = [c.source for c in nb.cells if c.cell_type == "code"]
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    definitions = []
    for source in codes:
        tree = ast.parse(source)
        definitions.extend(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    ns = dict(globals())
    ns.update(OPERATIONS=("add", "remove", "replace"), SEED=42, TARGET_PER_OPERATION=2,
              INFER_SIZE=518, MAX_IMAGE_SIDE=1024, MAX_ASPECT_RATIO_ERROR=.1,
              MASK_THRESHOLD=.5, MIN_COMPONENT_FRACTION=.001, MAX_MASK_FRACTION=.65,
              BOX_PADDING=.08, SCRIBBLE_VERSION="test",
              ANNOTATION_VERSION="v3-auto-group-qwen", PROMPT_VERSION="v3.1-qwen-grounded-style",
              MAX_AUTO_ASPECT_RATIO_ERROR=.05,
              TEXT_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct", text_revision="test-revision",
              AUTOMATION_SIGNATURE="test-policy", GROUP_POLICY_VERSION="test-groups",group_policy={},
              MERGE_OVERLAP_FRACTION=.2, MERGE_GAP_FRACTION=.025, MAX_AUTO_MERGE_EXPANSION=1.8,
              IGNORE_SECONDARY_FRACTION=.03, MAX_AUTO_CHANGE_FRACTION=.25, MAX_AUTO_BOX_FRACTION=.35,
              MAX_AUTO_BOX_TO_MASK=4, REJECT_CHANGE_FRACTION=.55, REJECT_ADD_CHANGE_FRACTION=.35,
              auto_reviews={},
              SCRIBBLE_STYLES=["box", "ellipse", "freehand_loop", "cross", "hatch", "short_strokes"],
              BASE_URL="https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/",
              RUN_NAME="pico_scribble_test", reviews={}, state={}, processing_spec={})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    # Real metadata schema and Flickr URL normalization across host variants.
    u1 = "https://farm7.staticflickr.com/8388/8590025416_9836bc771b_o.jpg"
    u2 = "http://c1.staticflickr.com/8/8388/8590025416_9836bc771b_o.jpg"
    assert ns["source_key"](u1) == ns["source_key"](u2) == "flickr:8590025416_9836bc771b_o.jpg"
    assert ns["operation_of"]("Remove an existing object") == "remove"
    assert ns["operation_of"]("Add new scene context/background") is None
    assert ns["operation_of"]("Relocate an object") is None
    rec = ns["normalize_record"](dict(edit_type="Add a new object to the scene", text="Add canoe.",
        summarized_text="Add a canoe.", open_image_input_url=u1, output_image="images/positive-edit/28821.png"))
    assert rec["id"] == "add_pico_28821" and rec["result_url"].endswith("/images/positive-edit/28821.png")

    # Every style is deterministic and visible, and interior styles do not paint outside mask.
    image = np.zeros((100, 140, 3), np.uint8)
    heat = np.zeros((100, 140), np.float32)
    heat[20:80, 25:115] = 1
    mask, boxes, flags = ns["mask_and_boxes"](heat, .5)
    assert len(boxes) == 1 and not flags
    rendered = set()
    for style in ns["SCRIBBLE_STYLES"]:
        a = np.array(ns["draw_scribble"](image, boxes, 42, style, mask))
        b = np.array(ns["draw_scribble"](image, boxes, 42, style, mask))
        assert np.array_equal(a, b) and np.any(a)
        if style in {"hatch", "short_strokes"}:
            assert not np.any(a[mask == 0])
        rendered.add(a.tobytes())
    assert len(rendered) == 6
    assert set(ns["allowed_styles"]("add")) == {"box", "ellipse", "freehand_loop"}
    assert "cross" in ns["allowed_styles"]("remove")
    # One click retains one component, not a separate reflection/shadow component.
    two_regions = np.zeros((100, 140), np.float32)
    two_regions[15:45, 20:60] = 1
    two_regions[65:85, 75:125] = 1
    cue, cue_boxes, cue_flags = ns["cue_from_click"](two_regions, .5, [30, 25])
    assert len(cue_boxes) == 1 and not cue_flags
    assert cue[25, 30] and not cue[75, 90]
    assert not ns["cue_from_click"](two_regions, .5, None)[1]
    assert not ns["cue_from_click"](two_regions, .5, [0, 0])[1]
    assert ns["event_to_point"](dict(dataX=21, dataY=33), 140, 100) == [21, 33]
    assert ns["event_to_point"](dict(relativeX=10, relativeY=25, boundingRectWidth=70, boundingRectHeight=50),140,100) == [20,50]
    add_rec = dict(id="add_pico_1", operation="add")
    remove_rec = dict(id="remove_pico_1", operation="remove")
    replace_rec = dict(id="replace_pico_1", operation="replace")
    assert ns["build_short_prompt"](add_rec,"ellipse","a canoe",reference="shape") == "Add a canoe inside the oval."
    assert ns["build_short_prompt"](remove_rec,"cross","the person",reference="shape") == "Remove the person marked with an X."
    assert "inside" not in ns["build_short_prompt"](remove_rec,"hatch","the person",reference="shape")
    assert ns["build_short_prompt"](replace_rec,"box","the telescope","a plant","generic") == "Replace the marked telescope with a plant."
    assert ns["prompt_reference"](add_rec) == ns["prompt_reference"](add_rec)
    # Group fragments, reject spread edits, and route ambiguity for manual review.
    simple = np.zeros((200, 200), np.float32); simple[70:110, 70:110] = 1
    assert ns["visual_assessment"](add_rec,simple)["status"] == "auto_accepted"
    broad = np.ones((200, 200), np.float32)
    assert ns["visual_assessment"](add_rec,broad)["status"] == "rejected"
    assert ns["visual_assessment"](add_rec,broad*0)["status"] == "rejected"
    split = simple.copy(); split[70:110, 111:121] = 1
    groups,metrics = ns["group_change_regions"](split,.5)
    assert metrics["component_count"]==2 and len(groups)==1
    selected,boxes,_ = ns["group_cue"](split,.5,0)
    assert len(boxes)==1 and selected[80,115] and not selected[80,110]
    distant = simple.copy(); distant[150:175,150:175]=1
    assert "multiple_significant_groups" in ns["visual_assessment"](add_rec,distant)["reasons"]
    chain = np.zeros((300,300),np.float32)
    for x in (20,41,62): chain[50:70,x:x+20]=1
    assert "merge_chain_expands_box" in ns["visual_assessment"](add_rec,chain)["reasons"]
    text_rec = dict(add_rec,prompt_original="Add a dark green canoe on the lake.")
    good = json.dumps(dict(operation="add",object="a dark green canoe",replacement="",needs_review=False))
    assert ns["parse_qwen_slots"](good,text_rec)["valid"]
    bad = good.replace("dark green", "purple")
    assert "ungrounded_prompt_words" in ns["parse_qwen_slots"](bad,text_rec)["reasons"]
    assert not ns["parse_qwen_slots"]("not JSON",text_rec)["valid"]
    possessive = dict(remove_rec,prompt_original="Remove the girl's red hat.")
    assert ns["parse_qwen_slots"](json.dumps(dict(operation="remove",object="the girl's red hat",replacement="",needs_review=False)),possessive)["valid"]
    quoted = dict(replace_rec,prompt_original="Replace the sign with a screen showing 'CLIMATE ACTION NOW'.")
    missing_quote=json.dumps(dict(operation="replace",object="the sign",replacement="a screen",needs_review=False))
    assert "quoted_target_text_missing" in ns["parse_qwen_slots"](missing_quote,quoted)["reasons"]
    distractor=dict(replace_rec,prompt_original="Replace the telescope by the window with a tall leafy potted plant.")
    wrong_dest=json.dumps(dict(operation="replace",object="the telescope",replacement="the window",needs_review=False))
    assert "replacement_not_in_destination_clause" in ns["parse_qwen_slots"](wrong_dest,distractor)["reasons"]
    aspect_case=dict(add_rec,aspect_ratio_error=.027,alignment_flags=["aspect_ratio_resized"])
    assert ns["visual_assessment"](aspect_case,simple)["status"]=="auto_accepted"
    aspect_case["aspect_ratio_error"]=.07
    assert ns["visual_assessment"](aspect_case,simple)["status"]=="needs_review"

    with tempfile.TemporaryDirectory(prefix="pico300-check-", dir=path.parent) as tmp:
        root = Path(tmp)
        ns["ROOT"] = root
        # Upload schema and traversal checks.
        source, target = root / "a.png", root / "b.png"
        Image.fromarray(image).save(source)
        target_image = image.copy(); target_image[mask > 0] = 180
        Image.fromarray(target_image).save(target)
        upload = root / "pairs.zip"
        with zipfile.ZipFile(upload, "w") as z:
            z.write(source, "original/a.png"); z.write(target, "result/a.png")
            z.writestr("pairs.jsonl", json.dumps(dict(operation="add", prompt_original="Add canoe.",
                prompt_short="Add canoe.", original_file_name="original/a.png", result_file_name="result/a.png")))
        imported = ns["load_upload_candidates"](upload)
        assert len(imported) == 1 and Path(imported[0]["local_original"]).is_file()
        evil = root / "evil.zip"
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("../escape.txt", "bad")
        try:
            ns["safe_unzip"](evil, root / "evil")
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe ZIP accepted")
        # Mock inference only: preparation, heatmap cache, source dedup all use real files.
        calls = []
        ns["infer_heatmap"] = lambda *args: (calls.append(1) or heat.copy())
        prepared = ns["prepare_one"](imported[0])
        assert prepared["status"] == "ready" and len(calls) == 1
        ns["prepare_one"](imported[0])
        assert len(calls) == 1, "Cache miss unnecessarily reran inference"
        ns["state"] = {prepared["id"]: prepared}
        other = {**imported[0], "id": "add_upload_duplicate"}
        try:
            ns["prepare_one"](other)
        except ValueError as exc:
            assert "Duplicate" in str(exc)
        else:
            raise AssertionError("Duplicate source accepted")
        # Fill 2 synthetic approved sources per operation, then export exact balanced quota.
        ns["state"], ns["reviews"] = {}, {}
        for op in ns["OPERATIONS"]:
            for i in range(2):
                uid = f"{op}_pico_{i}"
                rec = dict(id=uid, operation=op, source_id=str(i), status="ready", fingerprint=uid,
                    source_key=uid, source_hash=uid, source_url="https://example.org/"+uid,
                    result_url="https://example.org/result/"+uid, prompt_original="Do the edit.", alignment_flags=[])
                folder = ns["sample_dir"](rec); folder.mkdir(parents=True)
                shutil.copyfile(source, folder / "original.png"); shutil.copyfile(target, folder / "result.png")
                np.save(folder / "heatmap.npy", heat)
                rec["fingerprint"] = ns["config_signature"](dict(processing=ns["processing_spec"],
                    original=ns["digest_file"](folder / "original.png"), result=ns["digest_file"](folder / "result.png")))
                ns["state"][uid] = rec
                ns["reviews"][uid] = dict(status="manual_approved", prompt_short="Do the edit.", threshold=.5,
                    scribble_style="cross" if op == "remove" else "ellipse", flags_confirmed=True, fingerprint=rec["fingerprint"])
                review = ns["reviews"][uid]
                cue, _, _ = ns["cue_from_click"](heat, .5, [50, 50])
                review.update(cue_point=[50,50], cue_mask_hash=hashlib.sha256(cue.tobytes()).hexdigest(),
                    annotation_version=ns["ANNOTATION_VERSION"], prompt_version=ns["PROMPT_VERSION"],
                    automation_signature=ns["AUTOMATION_SIGNATURE"], selection_method="clicked_component",
                    prompt_object="the old object" if op=="replace" else "the object",
                    prompt_replacement="a new object" if op=="replace" else "",
                    prompt_reference="shape", prompt_override=False)
                review["prompt_short"] = ns["build_short_prompt"](rec,review["scribble_style"],review["prompt_object"],
                    review["prompt_replacement"],review["prompt_reference"])
        archive = ns["export_dataset"]()
        exported = Path(archive).with_suffix("")
        rows = [json.loads(line) for line in (exported / "metadata.jsonl").read_text().splitlines()]
        assert len(rows) == 6 and Counter(r["operation"] for r in rows) == dict(add=2, remove=2, replace=2)
        assert {r["scribble_style"] for r in rows} == {"ellipse", "cross"}
        assert all(r["review_status"] == "manual_approved" for r in rows)
        for r in rows:
            assert (exported / r["overlay_file_name"]).is_file()
            assert len(r["boxes"]) == 1 and r["cue_point"] == [50,50]
            assert (exported / r["change_mask_file_name"]).is_file()
            assert r["prompt_version"] == ns["PROMPT_VERSION"]
            if r["scribble_style"] == "cross":
                assert "marked with an X" in r["prompt_short"]
        # Fully automatic path: cached masks + mocked text generation, no manual approvals.
        saved_state=json.loads(json.dumps(ns["state"]))
        saved_manual=json.loads(json.dumps(ns["reviews"]))
        saved_prepare=ns["prepare_one"]
        auto_candidates=[]
        for op in ns["OPERATIONS"]:
            for index in range(3):
                auto_candidates.append(dict(id=f"{op}_pico_{index}",operation=op))
        small_heat=np.zeros_like(heat); small_heat[35:55,50:75]=1
        for rec in saved_state.values():
            np.save(ns["sample_dir"](rec)/"heatmap.npy",small_heat)
        def fake_prepare(candidate):
            rec=dict(saved_state[candidate["id"]])
            rec["prompt_original"]={"add":"Add the object.","remove":"Remove the object.",
                                    "replace":"Replace the object with a new object."}[rec["operation"]]
            return rec
        ns.update(state={},reviews={},auto_reviews={},candidates=auto_candidates,
                  prepare_one=fake_prepare,CANDIDATE_BATCH_PER_OPERATION=3)
        ns["extract_qwen_slots"] = lambda rec:dict(valid=True,reasons=[],object="the object",
            replacement="a new object" if rec["operation"]=="replace" else "",model=ns["TEXT_MODEL_ID"],
            revision=ns["text_revision"],cache_key="test-text-cache")
        ns["prepare_batch"]()
        assert len(ns["state"])==6, "Preparation must stop when automatic quota is reached"
        assert all(ns["current_review"](r)["status"]=="auto_accepted" for r in ns["state"].values())
        auto_archive=ns["export_dataset"]()
        auto_rows=[json.loads(line) for line in (Path(auto_archive).with_suffix("")/"metadata.jsonl").read_text().splitlines()]
        assert len(auto_rows)==6 and all(r["review_status"]=="auto_accepted" for r in auto_rows)
        assert all(r["selection_method"]=="auto_group" and len(r["boxes"])==1 for r in auto_rows)
        first=next(iter(ns["state"].values()))
        ns["reviews"][first["id"]]=dict(status="rejected",fingerprint=first["fingerprint"])
        assert ns["current_review"](first)["status"]=="rejected", "Manual rejection must override auto acceptance"
        ns.update(state=saved_state,reviews=saved_manual,auto_reviews={},prepare_one=saved_prepare)
        for rec in ns["state"].values():
            np.save(ns["sample_dir"](rec)/"heatmap.npy",heat)
        # If widget dependencies are installed, exercise actual callbacks without a browser/GPU.
        qa_deps = path.parents[1] / "output/notebook_qa_deps"
        if qa_deps.exists():
            sys.path.insert(0, str(qa_deps))
        try:
            import ipywidgets
            import ipyevents
        except ImportError:
            print("Widget callback checks skipped (optional ipywidgets/ipyevents not installed).")
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            saved_reviews = json.loads(json.dumps(ns["reviews"]))
            for review in ns["reviews"].values():
                review["status"] = "needs_review"
                review.pop("cue_point", None)
            ns.update(plt=plt, display=lambda *_: None, clear_output=lambda **_: None)
            gallery = next(s for s in codes if s.startswith("import ipywidgets as widgets"))
            exec(compile(gallery, str(path), "exec"), ns)
            selected_id = ns["sample_widget"].value
            assert ns["selected_point"] is None
            ns["handle_click"](dict(dataX=50, dataY=50))
            assert ns["selected_point"] == [50, 50]
            ns["object_widget"].value = "a canoe"
            ns["reference_widget"].value = "shape"
            ns["style_widget"].value = "ellipse"
            assert ns["prompt_widget"].value == "Add a canoe inside the oval."
            ns["save_decision"]("manual_approved")
            assert ns["reviews"][selected_id]["status"] == "manual_approved"
            assert ns["reviews"][selected_id]["cue_point"] == [50, 50]
            assert ns["sample_widget"].value != selected_id
            ns["manual_widget"].value = True
            ns["prompt_widget"].value = "An old manual prompt."
            new_style = "box" if ns["style_widget"].value != "box" else "ellipse"
            ns["style_widget"].value = new_style
            assert not ns["manual_widget"].value
            assert ns["prompt_widget"].value != "An old manual prompt."
            ns["handle_click"](dict(dataX=0, dataY=0))
            assert ns["selected_point"] is None
            pending_id = ns["sample_widget"].value
            ns["save_decision"]("manual_approved")
            assert ns["reviews"][pending_id]["status"] == "needs_review"
            ns["reviews"] = saved_reviews
            plt.close("all")
            print("Actual widget callbacks: click, prompt sync, approve/next, stale override reset passed.")
        # Pending/stale reviews can't fill quota; duplicates cannot be exported.
        uid = next(iter(ns["reviews"]))
        ns["reviews"][uid]["status"] = "needs_review"
        try:
            ns["export_dataset"]()
        except ValueError as exc:
            assert "thiếu" in str(exc)
        else:
            raise AssertionError("Pending pair filled approved quota")
        ns["reviews"][uid]["status"] = "manual_approved"
        ns["reviews"][uid]["fingerprint"] = "stale"
        assert ns["current_review"](ns["state"][uid]) == {}
    print("300-builder: syntax, mapping, ZIP, cache, dedup, six styles, reviews and export passed.")


if __name__ == "__main__":
    main()
