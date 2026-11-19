#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Install this checkout and its bundled SOL kernels before a worker command.
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
source_revision="$(git -C "$repo_root" rev-parse --short=12 HEAD)"
runtime_dir="${PHYSIQ_SOL_RUNTIME_DIR:-/tmp/physicsiq-sol-runtime}"
mkdir -p "$runtime_dir"
export UV_CACHE_DIR="$runtime_dir/uv-cache"
export PIP_CACHE_DIR="$runtime_dir/pip-cache"
export PYTHONDONTWRITEBYTECODE=1

uv venv --seed --system-site-packages --python "$(command -v python3)" "$runtime_dir/venv"
export VIRTUAL_ENV="$runtime_dir/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

python3 -m pip install \
    --constraint "$script_dir/public-runtime-constraints.txt" \
    --requirement "$script_dir/runtime-requirements.txt"
python3 -m pip install --no-deps "$repo_root/third_party/sol-attn"
VLLM_OMNI_TARGET_DEVICE=cuda \
VLLM_OMNI_VERSION_OVERRIDE="0.32.0rc2.dev101+g${source_revision}.sol" \
python3 -m pip install --no-deps "$repo_root"

python3 "$script_dir/preflight.py"
python3 -c 'from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline; print("Cosmos3 pipeline import passed")'
python3 -m pip freeze --all > "$runtime_dir/installed-packages.txt"
printf 'Runtime prepared from commit %s at %s\n' "$source_revision" "$runtime_dir"

if [[ "$#" -gt 0 ]]; then
    exec "$@"
fi
