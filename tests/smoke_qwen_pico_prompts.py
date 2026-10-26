"""Optional live Qwen smoke test; downloads the public 0.5B model on first run.

Run: python tests/smoke_qwen_pico_prompts.py
Only the text parser is tested, on CPU. No SIGMA downloads or image processing.
"""
import ast
import json
import re
from pathlib import Path

import nbformat
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root / "notebooks/build_pico_scribble_300_colab.ipynb", as_version=4)
    definitions = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.FunctionDef) and node.name in {"clean_slot", "parse_qwen_slots"}:
                definitions.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "QWEN_SYSTEM" for t in node.targets):
                definitions.append(node)
    ns = dict(re=re, json=json)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(root), "exec"), ns)
    cache = root / "output/qwen_smoke_cache"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = HfApi().model_info(model_id).sha
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, cache_dir=cache,
                                                torch_dtype=torch.float32).eval()
    records = [
        dict(operation="add", prompt_original="Add a dark green canoe in the lake."),
        dict(operation="remove", prompt_original="Remove the red flag and its white pole from the upper right of the image."),
        dict(operation="replace", prompt_original="Replace the telescope by the window with a tall leafy potted plant."),
    ]
    # Add held-out real instructions, not just the examples embedded in the system prompt.
    manifest = json.loads((root / "resource/flux2_klein_60_cases.json").read_text(encoding="utf-8"))
    for sample_id in ("34747", "39818", "196584"):
        row = next(row for row in manifest if row["output_image"].endswith("/"+sample_id+".png"))
        records.append(dict(operation=row["operation"].lower(), prompt_original=row["text"]))
    results = []
    for record in records:
        chat = tokenizer.apply_chat_template([
            dict(role="system", content=ns["QWEN_SYSTEM"]),
            dict(role="user", content=json.dumps(dict(operation=record["operation"], instruction=record["prompt_original"])))
        ], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=192, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        parsed = ns["parse_qwen_slots"](raw, record)
        results.append(dict(**record, **parsed))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    out = root / "output/qwen_smoke_results.json"
    out.write_text(json.dumps(dict(model=model_id, revision=revision, results=results), indent=2), encoding="utf-8")
    print("Valid:", sum(r["valid"] for r in results), "/", len(results), flush=True)


if __name__ == "__main__":
    main()
