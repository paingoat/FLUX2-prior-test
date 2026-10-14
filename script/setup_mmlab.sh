#!/usr/bin/env bash
# Lab setup for FLUX.2 [klein] 4B. Conda is already on the machine: this
# script creates the env, then installs the CUDA 12.8 PyTorch wheel, Python
# deps, both diffusers checkpoints, and Prior_test images.
# It does not download Miniconda or install a base environment.
# Safe to re-run. Weights go to HF_HOME (set this in .env; the RunPod
# /workspace default is not a lab path).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_ENV=0
SKIP_MODELS=0
SKIP_DATA=0

usage() {
  echo "Usage: bash script/setup_mmlab.sh [--skip-env] [--skip-models] [--skip-data]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-env|--skip-conda) SKIP_ENV=1 ;;
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

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
CONDA_ENV="${CONDA_ENV:-flux2-klein}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ "$HF_HOME" == /workspace/* && ! -d /workspace ]]; then
  echo "HF_HOME=$HF_HOME but /workspace does not exist on this machine." >&2
  echo "Set HF_HOME in .env to a directory on the lab disk, then re-run." >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$ROOT/output/gradio" "$ROOT/output/batch"
echo "HF_HOME=$HF_HOME"

conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    dirname "$(dirname "${CONDA_EXE}")"
    return
  fi
  local candidate
  for candidate in \
    "${HOME}/miniconda3" \
    "${HOME}/anaconda3" \
    "${HOME}/miniforge3" \
    "${HOME}/mambaforge" \
    /opt/conda \
    /opt/miniconda3 \
    /opt/anaconda3 \
    /usr/local/miniconda3 \
    /usr/local/anaconda3
  do
    if [[ -x "${candidate}/bin/conda" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "conda was not found. This script expects conda to be installed already." >&2
  echo "Open a shell where \`conda\` works, then re-run." >&2
  return 1
}

init_conda() {
  if [[ "${CONDA_READY:-0}" -eq 1 ]]; then
    return
  fi
  local base
  base="$(conda_base)" || exit 1
  if [[ ! -f "$base/etc/profile.d/conda.sh" ]]; then
    echo "conda at $base has no etc/profile.d/conda.sh." >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  set -u
  CONDA_READY=1
  echo "Using conda at $base"
}

activate_conda() {
  init_conda
  set +u
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Conda env $CONDA_ENV does not exist. Re-run without --skip-env." >&2
    exit 1
  fi
  conda activate "$CONDA_ENV"
  set -u
}

install_env() {
  init_conda
  set +u
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
    raise SystemExit("CUDA is not available. An RTX 5090 machine should show a GPU here.")
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
    echo "That file is not in git yet. Commit it, pull it on the machine, then re-run setup." >&2
    exit 1
  fi
  python "$ROOT/script/download_prior_test_data.py"
}

if [[ "$SKIP_ENV" -eq 0 ]]; then
  install_env
else
  echo "Skipping conda env and pip."
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
echo "  conda activate $CONDA_ENV"
