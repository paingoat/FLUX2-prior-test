"""Interactive FLUX.2 [klein] editor.

Compare an overlay alone with two references: clean original, then overlay.
Each run saves both source images, output, comparison and inference config.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.env import load_env

load_env()

import gradio as gr
from PIL import Image

from app.common.env import ROOT as REPO_ROOT
from app.common.env import slugify, timestamp
from app.common.flux2 import MODEL_SPECS, ModelManager
from app.common.visual import make_visual

OUTPUT_ROOT = REPO_ROOT / "output" / "gradio"
_MANAGER: ModelManager | None = None
MULTI_REFERENCE_INSTRUCTIONS = (
    "Image 1 is the original image to edit.\n"
    "Image 2 is the same image with scribble annotations indicating the edit.\n"
    "Use image 1 as the source and image 2 as spatial guidance.\n"
    "Do not reproduce the annotation strokes in the output.\n"
    "Preserve content outside the requested edit.\n\n"
    "Editing instruction: "
)


def effective_prompt(prompt: str, input_mode: str) -> str:
    if input_mode == "multi_reference":
        return MULTI_REFERENCE_INSTRUCTIONS + (prompt or "").strip()
    if input_mode == "overlay":
        return (prompt or "").strip()
    raise gr.Error(f"Unknown input mode: {input_mode!r}.")


def _manager() -> ModelManager:
    if _MANAGER is None:
        raise RuntimeError("Model manager is not initialized.")
    return _MANAGER


def _editor() -> gr.ImageEditor:
    kwargs = dict(
        label="Image and scribble",
        type="pil",
        sources=["upload", "clipboard"],
        image_mode="RGB",
        brush=gr.Brush(
            colors=["#FF0000", "#00FF00", "#0000FF", "#FFFF00"],
            default_color="#FF0000",
            color_mode="fixed",
            default_size=8,
        ),
        eraser=gr.Eraser(default_size=16),
        height=520,
    )
    if "transforms" in inspect.signature(gr.ImageEditor.__init__).parameters:
        kwargs["transforms"] = []
    return gr.ImageEditor(**kwargs)


def _flatten(editor_value) -> Image.Image:
    if editor_value is None:
        raise gr.Error("Upload an image first.")
    if isinstance(editor_value, Image.Image):
        return _to_rgb(editor_value)

    if not isinstance(editor_value, dict):
        raise gr.Error(f"Unexpected editor value ({type(editor_value).__name__}).")

    background = _as_image(editor_value.get("background"))
    composite = _as_image(editor_value.get("composite"))
    if composite is None and background is not None:
        composite = background.copy()
        for layer in editor_value.get("layers") or []:
            overlay = _as_image(layer)
            if overlay is None:
                continue
            if overlay.size != composite.size:
                overlay = overlay.resize(composite.size, Image.Resampling.LANCZOS)
            if overlay.mode == "RGBA":
                base = composite.convert("RGBA")
                base.alpha_composite(overlay)
                composite = base
            else:
                composite.paste(overlay)
    if composite is None:
        raise gr.Error("Upload an image first.")
    if background is not None and composite.size != background.size:
        composite = composite.resize(background.size, Image.Resampling.LANCZOS)
    return _to_rgb(composite)


def _editor_images(editor_value) -> tuple[Image.Image, Image.Image]:
    overlay = _flatten(editor_value)
    original = (
        _as_image(editor_value.get("background"))
        if isinstance(editor_value, dict)
        else _as_image(editor_value)
    )
    if original is None:
        raise gr.Error("The clean original is missing. Upload the original image again.")
    original = _to_rgb(original)
    if overlay.size != original.size:
        overlay = overlay.resize(original.size, Image.Resampling.LANCZOS)
    return original, overlay


def _as_image(value) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    try:
        return Image.fromarray(value)
    except Exception as exc:
        raise gr.Error(f"Could not read the editor image: {exc}") from exc


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.getchannel("A"))
        return flat
    return image.convert("RGB")


def _apply_variant(variant: str):
    spec = MODEL_SPECS[variant]
    return (
        gr.update(value=spec.steps, interactive=not spec.fixed),
        gr.update(value=spec.guidance, interactive=not spec.fixed),
    )


def run_edit(name, prompt, editor_value, variant, seed, steps, guidance, input_mode="overlay"):
    cleaned_name = (name or "").strip()
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_name:
        raise gr.Error("Name is required.")
    if not cleaned_prompt:
        raise gr.Error("Prompt is required.")
    if seed is None:
        raise gr.Error("Seed is required.")

    model_prompt = effective_prompt(cleaned_prompt, input_mode)
    original, overlay = _editor_images(editor_value)
    source = [original, overlay] if input_mode == "multi_reference" else overlay
    result, meta = _manager().edit(
        variant=variant,
        image=source,
        prompt=model_prompt,
        seed=int(seed),
        steps=None if steps is None else int(steps),
        guidance=None if guidance is None else float(guidance),
    )
    visual = make_visual(overlay, result, f"{input_mode}\n{model_prompt}", original=original)

    run_dir = OUTPUT_ROOT / f"{slugify(cleaned_name)}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    original.save(run_dir / "original.png")
    overlay.save(run_dir / "overlay.png")
    result.save(run_dir / "result.png")
    visual.save(run_dir / "visual.png")
    config = {
        "name": cleaned_name,
        "prompt": cleaned_prompt,
        "effective_prompt": model_prompt,
        "input_mode": input_mode,
        "reference_order": ["original.png", "overlay.png"] if input_mode == "multi_reference" else ["overlay.png"],
        "files": {
            "original": "original.png",
            "overlay": "overlay.png",
            "input": "overlay.png",
            "result": "result.png",
            "visual": "visual.png",
        },
        "run": meta,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result, visual, config, str(run_dir)


def build_demo() -> gr.Blocks:
    distill = MODEL_SPECS["distill"]
    default_seed = int(os.environ.get("SEED", "42"))
    with gr.Blocks(title="FLUX.2 klein edit") as demo:
        gr.Markdown(
            "# FLUX.2 [klein] 4B edit\n"
            "Draw on a clean uploaded image. Compare overlay-only editing with "
            "multi-reference editing (original first, annotated image second). "
            "Distilled settings are fixed at "
            f"**{distill.steps} steps, guidance {distill.guidance:g}**. "
            "Base defaults are **50 steps, guidance 4.0** and can be changed."
        )
        with gr.Row():
            with gr.Column():
                name = gr.Textbox(label="Name", placeholder="experiment name")
                prompt = gr.Textbox(label="Prompt", lines=4)
                editor = _editor()
                input_mode = gr.Radio(
                    choices=[
                        ("Overlay only (baseline)", "overlay"),
                        ("Multi-reference (original + overlay)", "multi_reference"),
                    ],
                    value="overlay",
                    label="Input mode",
                )
                prompt_preview = gr.Textbox(label="Prompt sent to model", lines=8, interactive=False)
                variant = gr.Radio(
                    choices=[
                        ("distill — 4 steps, guidance 1.0 (fixed)", "distill"),
                        ("base — 50 steps, guidance 4.0", "base"),
                    ],
                    value="distill",
                    label="Model",
                )
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed", value=default_seed, precision=0)
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=distill.steps,
                        interactive=False,
                    )
                    guidance = gr.Slider(
                        label="Guidance",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=distill.guidance,
                        interactive=False,
                    )
                run_btn = gr.Button("Run", variant="primary")
            with gr.Column():
                result = gr.Image(label="Result", type="pil")
                visual = gr.Image(label="Visual", type="pil")
                config = gr.JSON(label="Config")
                saved = gr.Textbox(label="Saved to")

        variant.change(_apply_variant, inputs=variant, outputs=[steps, guidance])
        prompt.change(effective_prompt, inputs=[prompt, input_mode], outputs=prompt_preview)
        input_mode.change(effective_prompt, inputs=[prompt, input_mode], outputs=prompt_preview)
        run_btn.click(
            run_edit,
            inputs=[name, prompt, editor, variant, seed, steps, guidance, input_mode],
            outputs=[result, visual, config, saved],
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    global _MANAGER
    parser = argparse.ArgumentParser(description="Gradio editor for FLUX.2 [klein] 4B.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-share", action="store_true", help="Do not request a public Gradio link.")
    parser.add_argument("--preload", choices=["all", "distill", "base"], default="all")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="CPU-offload a single pipeline. Use this below 24 GB VRAM; model switches reload.",
    )
    args = parser.parse_args()

    port = args.port if args.port is not None else int(os.environ.get("GRADIO_PORT", "7860"))
    share_env = os.environ.get("GRADIO_SHARE", "1").strip().lower()
    share = share_env not in {"0", "false", "no"} and not args.no_share

    _MANAGER = ModelManager(offload=args.offload)
    if args.offload and args.preload == "all":
        print("Offload keeps one pipeline resident; preloading distill only. Base loads on first use.")
        _MANAGER.preload("distill")
    else:
        print(f"Loading {args.preload}...")
        _MANAGER.preload(args.preload)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_demo()
    if callable(getattr(demo, "queue", None)):
        demo.queue()
    demo.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
