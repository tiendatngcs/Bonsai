#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-bonsai:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-bonsai-dev}"
USE_GPU="${USE_GPU:-1}"
WORKDIR_IN_CONTAINER="/workspace/Bonsai"

source "${SCRIPT_DIR}/common.sh"

echo "Using Docker image: ${IMAGE_NAME}"
echo "Container name: ${CONTAINER_NAME}"

require_docker_access

mkdir -p "${ROOT_DIR}/traces" "${ROOT_DIR}/data"

docker_args=(
  --rm
  -it
  --name "${CONTAINER_NAME}"
  -v "${ROOT_DIR}:${WORKDIR_IN_CONTAINER}"
  -w "${WORKDIR_IN_CONTAINER}"
)

if [[ "${USE_GPU}" == "1" ]]; then
    docker_args+=(
      --gpus all
      -e NVIDIA_VISIBLE_DEVICES=all
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    )
fi

if [[ -n "${GUROBI_HOME_HOST:-}" ]]; then
    docker_args+=(
      -v "${GUROBI_HOME_HOST}:/opt/gurobi/linux64:ro"
      -e GUROBI_HOME=/opt/gurobi/linux64
    )
fi

if [[ -n "${GUROBI_LICENSE_HOST:-}" ]]; then
    docker_args+=(
      -v "${GUROBI_LICENSE_HOST}:/opt/gurobi/gurobi.lic:ro"
      -e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
    )
fi

if [[ "$#" -gt 0 ]]; then
    command=("$@")
else
    command=(bash)
fi

docker run "${docker_args[@]}" "${IMAGE_NAME}" "${command[@]}"

# usage

# IMAGE_NAME=tiendatngcs/bonsai_exec_only:dev source ./docker/run_container.sh