# FLUX.2 [klein] 4B edit tests

Small experiment harness for image editing with Black Forest Labs' FLUX.2 [klein] 4B. A Gradio app draws a scribble on an uploaded photo and edits it. A batch CLI then runs the same editor over two datasets: VIBE (the scribble is already on the image) and Prior (text prompt only, no relocate step).

Inference uses diffusers `Flux2KleinPipeline` and the sampling settings published in the official [flux2](https://github.com/black-forest-labs/flux2) repo.

| Variant | Repo | Steps | Guidance | Notes |
| --- | --- | --- | --- | --- |
| distill | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 4 | 1.0 | Fixed. The checkpoint is step- and guidance-distilled, so guidance is ignored. |
| base | [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | 50 | 4.0 | Undistilled. Classifier-free guidance with an empty negative prompt. |

Reference images larger than 1024×1024 are scaled down to that area. Each side is rounded down to a multiple of 16. The output matches that size. Both checkpoints share one Qwen3-4B text encoder and one VAE, so the apps load those once.

## Layout

```
app/gradio_inference/app.py     interactive editor
app/batch_inference/run_batch.py
app/results_gallery/server.py   batch results gallery
app/common/                     pipeline, env, comparison image
data/VIBE_test/<operation>/     add, relocate, remove, replace
data/Prior_test/<operation>/<case>/
script/setup_runpod.sh
script/setup_mmlab.sh          lab machine; conda already installed
script/download_prior_test_data.py
output/gradio/                  one folder per Gradio run
output/batch/                   one tree per model and batch run
```

`output/` and `.env` are gitignored.

## Quick start

On the RunPod pod (the setup script is Linux-only):

```bash
cp .env.example .env
# optional: set HF_TOKEN in .env
bash script/setup_runpod.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate flux2-klein
```

`setup_runpod.sh` installs Miniconda under `CONDA_DIR`, accepts the Anaconda channel terms, creates the `flux2-klein` env, installs PyTorch 2.8.0 from the CUDA 12.8 wheel index, installs `requirements.txt`, checks that the wheel lists `sm_120`, downloads both diffusers checkpoints into `HF_HOME`, and downloads Prior_test images.

Prior_test download needs `resource/flux2_klein_60_cases.json` in the checkout. Commit that file before cloning the repo on the pod. VIBE_test is already in git.

Re-running the script skips work that is already done. Useful flags: `--skip-conda`, `--skip-models`, `--skip-data`.

On a lab machine that already has conda, use `script/setup_mmlab.sh` instead. It does not download Miniconda. It creates the `flux2-klein` env with the conda already on the machine, then installs the same PyTorch wheel, Python deps, checkpoints, and Prior_test images. Set `HF_HOME` in `.env` to a directory on that machine before running it. The script ignores `CONDA_DIR`.

```bash
cp .env.example .env
# set HF_HOME to a local disk path; optional: set HF_TOKEN
bash script/setup_mmlab.sh
conda activate flux2-klein
```

It finds conda on `PATH`, then via `CONDA_EXE`, then under the usual prefixes (`~/miniconda3`, `~/anaconda3`, `~/miniforge3`, `/opt/conda`). Flags: `--skip-env`, `--skip-models`, `--skip-data`.

## Gradio

Hướng dẫn tiếng Việt: [test 10 ảnh Prior với prompt ngắn, tự lưu và upload Hugging Face](docs/gradio_scribble_test_vi.md).

```bash
python app/gradio_inference/app.py
```

The process prints a local URL and a public `*.gradio.live` link (`GRADIO_SHARE=1` in `.env`). Pass `--no-share` to stay local. The server listens on `0.0.0.0` and `GRADIO_PORT` (default 7860).

The form is a name, a prompt, an image editor (upload plus a red/green/blue/yellow scribble), an input mode, and a model switch. Distill locks steps and guidance. Base starts at 50 / 4.0 and can be changed under Advanced. Both models load at startup and share the text encoder and VAE, so switching is immediate.

Upload a **clean original**, then draw the annotations in the editor. Choose:

- **Overlay only (baseline)**: sends the annotated composite as `image=overlay_image`, with your prompt unchanged.
- **Multi-reference (original + overlay)**: sends `image=[original_image, overlay_image]`. A prompt prefix explains that image 1 is the source and image 2 contains spatial editing guidance, and asks to remove annotation strokes and preserve surrounding content. The full prompt appears in **Prompt sent to model**.

Both modes use output dimensions derived from the original (at most 1024×1024 pixels in area, sides rounded down to multiples of 16). Separate references and prompting do not guarantee scribble compliance; this experiment measures it. Uploading an image that already has strokes does not provide a clean original.

Each successful click writes `output/gradio/<name>_<YYYYmmdd-HHMMSS>_<unique-id>/`:

- `original.png` — clean editor background
- `overlay.png` — original with the scribble flattened on top
- `result.png`
- `visual.png` — original, overlay and result side by side, mode and effective prompt underneath
- `config.json` — input mode, ordered reference filenames, user/effective prompts, seed, steps, guidance, reference sizes, timing, peak allocated VRAM, torch / diffusers versions, GPU name
- `metadata.jsonl` — one ImageFolder row with original/overlay/result/visual image columns and ordered references, ready for Hugging Face dataset upload

A ZIP of these six files is also saved under `output/gradio_exports/` and returned in **Download run (images + metadata ZIP)**. Upload either selected ZIPs or the run folders to a Hugging Face dataset repository; keep one copy per run. Upload remains manual. Existing runs are not backfilled. All output remains gitignored.

For an A/B run, keep the image, scribble, editing instruction, model, seed, steps and guidance unchanged; run once per input mode using distinct experiment names. Repeat with distill and base on add, remove, replace and relocate examples. Compare instruction accuracy, scribble localization, stroke removal, preservation outside the edit, elapsed time and `peak_vram_allocated_mb`. The VRAM number includes resident model weights and allocations during inference, not total device memory or reserved memory; use the same preload/offload settings for comparisons. Two references require more memory. The multi-reference mode deliberately adds role instructions, so this compares the complete input-and-prompt strategies rather than isolating image count alone.

Batch inference remains single-image. Multi-reference batch evaluation requires paired clean/annotated inputs; do not substitute target/ground-truth images for clean inputs.

CPU wiring checks (no model downloads or CUDA needed):

```bash
python -m unittest discover -s tests -v
```

```bash
python app/gradio_inference/app.py --preload distill   # one model
python app/gradio_inference/app.py --offload           # one pipeline, CPU offload
```

## Batch

```bash
python app/batch_inference/run_batch.py
python app/batch_inference/run_batch.py --models distill --phases vibe --limit 2
python app/batch_inference/run_batch.py --run-name run-20261014-150000 --offload
```

Phase `vibe` reads `data/VIBE_test/<operation>/metadata.json` and sends each image unchanged (the cue is already drawn). Phase `prior` reads `data/Prior_test/<operation>/<case>/metadata.json`, uses the sample `text` as the prompt, and does not run a relocate operation. Models run one after another. The default seed is `SEED` from `.env` (42).

A tqdm bar counts images for the current model. The log prints a start and done line for each phase, operation, and Prior case, with ok / error / skipped counts and elapsed time. One failed image is recorded and the run continues.

Running the same `--run-name` again skips an image that already has a non-empty file and an `"ok"` record.

```
output/batch/<flux2-klein-distill|flux2-klein-base>/run-<timestamp>/
  run_config.json
  VIBE/<operation>/results/<id>.png
  VIBE/<operation>/results.json
  Prior/<operation>/<case>/results/output_<stem>.png
  Prior/<operation>/<case>/results.json
```

`results.json` is rewritten after every new image. Paths inside it are relative to the repo (inputs) or to the case folder (`results/<file>`).

## Results gallery

A local viewer lists every VIBE and Prior sample and, when present, the batch PNG for the selected model and run. Missing results show an empty frame. On Prior only, an eye button on the output pane toggles ground truth in place of the model output. Like / dislike / comment controls write to `app/results_gallery/feedback.json`.

```bash
python app/results_gallery/server.py
python app/results_gallery/server.py --port 8503
```

Open http://127.0.0.1:8502 (port 8502 so it can run next to the Prior input viewer on 8501). From a laptop:

```bash
ssh -L 8502:localhost:8502 user@host
```

The catalog is built from metadata even when `output/batch` is still empty. Click **Reload** after a batch run writes new images. JPEG thumbnails go in `app/results_gallery/.cache/` (Pillow; without Pillow the thumb route serves the original file).

## RTX 5090

Use the cu128 PyTorch wheel from the setup script (`torch==2.8.0`). The stock PyPI wheel has no `sm_120` kernels. Do not install flash-attn or xformers; diffusers uses PyTorch SDPA.

Both transformers plus the shared text encoder and VAE are about 24 GB of bf16 weights, which fits a 32 GB card with both models resident. If a base run (50 steps, classifier-free guidance) runs out of memory, start Gradio with `--offload` or batch with `--offload`. Offload keeps a single pipeline loaded and reloads it on a model switch.
