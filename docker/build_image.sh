#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-bonsai:dev}"
PYTORCH_OPTION="${PYTORCH_OPTION:-bonsai}"
MAX_JOBS="${MAX_JOBS:-4}"
DOCKER_PROGRESS="${DOCKER_PROGRESS:-plain}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"

source "${SCRIPT_DIR}/common.sh"

echo "Docker image: ${IMAGE_NAME}"
echo "Using PyTorch option: ${PYTORCH_OPTION}"

require_docker_access

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to validate PYTORCH_OPTION against pytorch_compatibility.json."
    exit 1
fi

if ! jq -e ".\"${PYTORCH_OPTION}\"" "${ROOT_DIR}/pytorch_compatibility.json" >/dev/null; then
    valid_options="$(jq -r 'keys | join(", ")' "${ROOT_DIR}/pytorch_compatibility.json")"
    echo "Unknown PYTORCH_OPTION '${PYTORCH_OPTION}'. Valid options: ${valid_options}"
    exit 1
fi

echo "Building Docker image '${IMAGE_NAME}' with profile '${PYTORCH_OPTION}' (MAX_JOBS=${MAX_JOBS}, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})"

if [[ "${PYTORCH_OPTION}" == "bonsai" ]]; then
    cat <<EOF
Note: the 'bonsai' profile builds a patched PyTorch from source inside Docker.
This is significantly slower and more resource-intensive than the prebuilt 'bonsai_exec_only' profile.
The Docker installer mirrors the Conda source-build flow, including the pinned
CMake 3.31.1 toolchain and non-isolated editable installs for PyTorch/TorchVision.
It also pins TORCH_CUDA_ARCH_LIST to a PyTorch-2.3-compatible set by default so
newer names like 9.0a do not break the build.
For a faster smoke-test image, run:
  PYTORCH_OPTION=bonsai_exec_only source ./docker/build_image.sh
EOF
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

docker build \
  --progress "${DOCKER_PROGRESS}" \
  --build-arg PYTORCH_OPTION="${PYTORCH_OPTION}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  --tag "${IMAGE_NAME}" \
  "${ROOT_DIR}"


# Usage

# IMAGE_NAME=tiendatngcs/bonsai-exec-only:dev PYTORCH_OPTION=bonsai_exec_only MAX_JOBS=4 source ./docker/build_image.sh
