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
app/common/                     pipeline, env, comparison image
data/VIBE_test/<operation>/     add, relocate, remove, replace
data/Prior_test/<operation>/<case>/
script/setup_runpod.sh
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

## Gradio

```bash
python app/gradio_inference/app.py
```

The process prints a local URL and a public `*.gradio.live` link (`GRADIO_SHARE=1` in `.env`). Pass `--no-share` to stay local. The server listens on `0.0.0.0` and `GRADIO_PORT` (default 7860).

The form is a name, a prompt, an image editor (upload plus a red/green/blue/yellow scribble), and a model switch. Distill locks steps and guidance. Base starts at 50 / 4.0 and can be changed under Advanced. Both models load at startup and share the text encoder and VAE, so switching is immediate.

Each click writes `output/gradio/<name>_<YYYYmmdd-HHMMSS>/`:

- `input.png` — uploaded image with the scribble flattened on top (this is what the model sees)
- `result.png`
- `visual.png` — input and result side by side, prompt caption underneath
- `config.json` — prompt, seed, steps, guidance, sizes, timing, torch / diffusers versions, GPU name

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

## RTX 5090

Use the cu128 PyTorch wheel from the setup script (`torch==2.8.0`). The stock PyPI wheel has no `sm_120` kernels. Do not install flash-attn or xformers; diffusers uses PyTorch SDPA.

Both transformers plus the shared text encoder and VAE are about 24 GB of bf16 weights, which fits a 32 GB card with both models resident. If a base run (50 steps, classifier-free guidance) runs out of memory, start Gradio with `--offload` or batch with `--offload`. Offload keeps a single pipeline loaded and reloads it on a model switch.
