#!/usr/bin/env bash
# Pod setup for FLUX.2 [klein] 4B: Miniconda, CUDA 12.8 PyTorch (sm_120),
# Python deps, both diffusers checkpoints, and Prior_test images.
# Safe to re-run. Weights go to HF_HOME (default /workspace/data/hf_cache).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_CONDA=0
SKIP_MODELS=0
SKIP_DATA=0

usage() {
  echo "Usage: bash script/setup_runpod.sh [--skip-conda] [--skip-models] [--skip-data]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-conda) SKIP_CONDA=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ ! -f .env ]]; then
  echo "Missing .env. Run: cp .env.example .env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source <(tr -d '\r' < .env)
set +a

HF_HOME="${HF_HOME:-/workspace/data/hf_cache}"
CONDA_DIR="${CONDA_DIR:-/workspace/miniconda3}"
CONDA_ENV="${CONDA_ENV:-flux2-klein}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

mkdir -p "$HF_HOME" "$ROOT/output/gradio" "$ROOT/output/batch"
echo "HF_HOME=$HF_HOME"

activate_conda() {
  if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    echo "Miniconda is not installed at $CONDA_DIR. Re-run without --skip-conda." >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  set -u
}

install_conda() {
  if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    echo "Installing Miniconda into $CONDA_DIR"
    installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
    curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$installer" -b -p "$CONDA_DIR"
    rm -f "$installer"
  else
    echo "Miniconda already present at $CONDA_DIR"
  fi

  set +u
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Creating conda env $CONDA_ENV (Python 3.12)"
    conda create -y -n "$CONDA_ENV" python=3.12
  else
    echo "Conda env $CONDA_ENV already exists"
  fi
  conda activate "$CONDA_ENV"
  set -u

  python -m pip install --upgrade pip
  python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -r "$ROOT/requirements.txt"

  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. An RTX 5090 pod should show a GPU here.")
print("gpu", torch.cuda.get_device_name(0))
archs = list(torch.cuda.get_arch_list())
print("arch", archs)
if not any(item in {"sm_120", "compute_120"} for item in archs):
    raise SystemExit(
        "This PyTorch build does not list sm_120. RTX 5090 needs the cu128 wheel (torch==2.8.0)."
    )
PY
}

download_models() {
  activate_conda
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "$HF_TOKEN"
  else
    echo "HF_TOKEN is empty. Downloading the public FLUX.2 [klein] 4B repos anonymously."
  fi
  local repo
  for repo in black-forest-labs/FLUX.2-klein-4B black-forest-labs/FLUX.2-klein-base-4B; do
    echo "Downloading $repo (diffusers folders only; skipping the single-file weights and preview jpgs)"
    hf download "$repo" \
      model_index.json \
      scheduler/ \
      text_encoder/ \
      tokenizer/ \
      transformer/ \
      vae/
  done
}

download_data() {
  activate_conda
  if [[ ! -f "$ROOT/resource/flux2_klein_60_cases.json" ]]; then
    echo "Missing resource/flux2_klein_60_cases.json." >&2
    echo "That file is not in git yet. Commit it, pull it on the pod, then re-run setup." >&2
    exit 1
  fi
  python "$ROOT/script/download_prior_test_data.py"
}

if [[ "$SKIP_CONDA" -eq 0 ]]; then
  install_conda
else
  echo "Skipping conda and pip."
fi

if [[ "$SKIP_MODELS" -eq 0 ]]; then
  download_models
else
  echo "Skipping model download."
fi

if [[ "$SKIP_DATA" -eq 0 ]]; then
  download_data
else
  echo "Skipping Prior_test download. data/VIBE_test is already in the repo."
fi

echo "Setup finished. Activate the env with:"
echo "  source \"$CONDA_DIR/etc/profile.d/conda.sh\" && conda activate $CONDA_ENV"
