#!/usr/bin/env bash
set -euo pipefail

OPTION="${1:-bonsai}"
ROOT_DIR="${BONSAI_ROOT:-}"
if [[ -z "${ROOT_DIR}" ]]; then
    ROOT_DIR="$(pwd)"
fi
CONFIG_FILE="${ROOT_DIR}/pytorch_compatibility.json"
SRC_DIR="/opt/src"
VISION_DIR="${SRC_DIR}/vision"

extract_repo_name() {
    local repo_url="$1"
    local repo_name_with_git="${repo_url##*/}"
    local repo_name="${repo_name_with_git%.git}"
    echo "${repo_name}"
}

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to parse ${CONFIG_FILE}."
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Could not find ${CONFIG_FILE}. Set BONSAI_ROOT to the repository root."
    exit 1
fi

if ! jq -e ".\"${OPTION}\"" "${CONFIG_FILE}" >/dev/null; then
    valid_options="$(jq -r 'keys | join(", ")' "${CONFIG_FILE}")"
    echo "Unknown compatibility option '${OPTION}'. Valid options: ${valid_options}"
    exit 1
fi

PYTORCH_TYPE="$(jq -r ".\"${OPTION}\".pytorch_type" "${CONFIG_FILE}")"
PYTORCH_REPO="$(jq -r ".\"${OPTION}\".pytorch_repo" "${CONFIG_FILE}")"
PYTORCH_VERSION="$(jq -r ".\"${OPTION}\".pytorch_version" "${CONFIG_FILE}")"
TORCHVISION_VERSION="$(jq -r ".\"${OPTION}\".torchvision_version" "${CONFIG_FILE}")"
PYTHON_VERSION="$(jq -r ".\"${OPTION}\".python_version" "${CONFIG_FILE}")"
TRANSFORMERS_VERSION="$(jq -r ".\"${OPTION}\".transformers_version" "${CONFIG_FILE}")"
PATCH_FILE="$(jq -r ".\"${OPTION}\".patch_file" "${CONFIG_FILE}")"
PYTORCH_WHEEL_INDEX_URL="${PYTORCH_WHEEL_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"

if [[ "${PYTORCH_VERSION}" == "null" || "${TORCHVISION_VERSION}" == "null" || "${PYTHON_VERSION}" == "null" || "${TRANSFORMERS_VERSION}" == "null" ]]; then
    echo "Invalid compatibility option: ${OPTION}"
    exit 1
fi

echo "Installing Bonsai Docker environment for profile '${OPTION}'"
echo "Python ${PYTHON_VERSION}, PyTorch ${PYTORCH_VERSION}, TorchVision ${TORCHVISION_VERSION}, Transformers ${TRANSFORMERS_VERSION}"
echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

mkdir -p "${SRC_DIR}"

# Mirror the Conda source-build helpers: keep CMake on 3.x and install the
# Python-side build tools in the runtime environment used for editable builds.
pip install \
    "cmake==3.31.1" \
    filelock \
    jinja2 \
    networkx \
    ninja \
    numpy \
    pyyaml \
    sympy \
    tensorboard \
    tqdm \
    typing_extensions \
    datasets

case "${PYTORCH_TYPE}" in
    from_source)
        if [[ "${PYTORCH_REPO}" == "null" ]]; then
            echo "A PyTorch repository is required for profile '${OPTION}'."
            exit 1
        fi

        if [[ "${PATCH_FILE}" != "null" && ! -f "${ROOT_DIR}/${PATCH_FILE}" ]]; then
            echo "Patch file '${ROOT_DIR}/${PATCH_FILE}' does not exist."
            exit 1
        fi

        PYTORCH_REPO_NAME="$(extract_repo_name "${PYTORCH_REPO}")"
        PYTORCH_DIR="${SRC_DIR}/${PYTORCH_REPO_NAME}"

        echo "Cloning PyTorch ${PYTORCH_VERSION} from ${PYTORCH_REPO}"
        git clone --branch "v${PYTORCH_VERSION}" --depth 1 --recursive "${PYTORCH_REPO}" "${PYTORCH_DIR}"
        pushd "${PYTORCH_DIR}" >/dev/null

        if [[ "${PATCH_FILE}" != "null" ]]; then
            echo "Applying patch ${PATCH_FILE}"
            git apply "${ROOT_DIR}/${PATCH_FILE}"
        fi

        git submodule sync
        git submodule update --init --recursive

        export USE_NUMPY=1
        export PYTORCH_HOME="${PYTORCH_DIR}"
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export CPATH="${CUDA_HOME}/include:${CPATH:-}"
        export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
        export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
        export TORCH_CUDA_ARCH_LIST
        export BUILD_TEST=0
        export USE_CUDA=1
        export USE_CUDNN=1
        export USE_NCCL=0
        GCC_WARNING_WORKAROUNDS="-Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-maybe-uninitialized -Wno-uninitialized"
        export CFLAGS="${CFLAGS:-} ${GCC_WARNING_WORKAROUNDS}"
        export CXXFLAGS="${CXXFLAGS:-} ${GCC_WARNING_WORKAROUNDS}"
        export CMAKE_C_FLAGS="${CMAKE_C_FLAGS:-} ${GCC_WARNING_WORKAROUNDS}"
        export CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-} ${GCC_WARNING_WORKAROUNDS}"

        echo "Building and installing patched PyTorch from source (MAX_JOBS=${MAX_JOBS:-unset})"
        pip install --no-build-isolation -e .
        popd >/dev/null

        echo "Cloning TorchVision ${TORCHVISION_VERSION}"
        git clone --branch "v${TORCHVISION_VERSION}" --depth 1 https://github.com/pytorch/vision.git "${VISION_DIR}"
        pushd "${VISION_DIR}" >/dev/null
        export FORCE_CUDA=1
        export LD_LIBRARY_PATH="${PYTORCH_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
        pip install "setuptools<81"
        pip install --no-build-isolation -e .
        popd >/dev/null

        pip install "transformers==${TRANSFORMERS_VERSION}"
        ;;
    prebuilt|pip_prebuilt)
        echo "Installing prebuilt CUDA-enabled PyTorch and TorchVision wheels from ${PYTORCH_WHEEL_INDEX_URL}"
        pip install \
            "torch==${PYTORCH_VERSION}" \
            "torchvision==${TORCHVISION_VERSION}" \
            --index-url "${PYTORCH_WHEEL_INDEX_URL}"
        pip install "transformers==${TRANSFORMERS_VERSION}"
        ;;
    *)
        echo "Unsupported pytorch_type '${PYTORCH_TYPE}'"
        exit 1
        ;;
esac

pushd "${ROOT_DIR}" >/dev/null
pip install -e ./rockmate-bonsai/rkgb -e ./rockmate-bonsai/rockmate
popd >/dev/null

echo "Bonsai Docker environment installation finished."
