FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_OPTION=bonsai
ARG MAX_JOBS=4
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

ENV VIRTUAL_ENV=/opt/venv
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
ENV CPATH="${CUDA_HOME}/include:${CPATH}"
ENV CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}"
ENV LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CC=gcc-12
ENV CXX=g++-12
ENV MAX_JOBS=${MAX_JOBS}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV BONSAI_ROOT=/workspace/Bonsai
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    gcc-12 \
    g++-12 \
    gfortran \
    git \
    jq \
    libbz2-dev \
    libffi-dev \
    libjpeg-dev \
    liblzma-dev \
    libomp-dev \
    libopenblas-dev \
    libpng-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    ninja-build \
    pkg-config \
    software-properties-common \
    xz-utils \
    zlib1g-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/Bonsai
COPY . /workspace/Bonsai

RUN install -m 0755 docker/install_bonsai.sh /usr/local/bin/install_bonsai.sh && \
    install -m 0755 docker/container_entrypoint.sh /usr/local/bin/container_entrypoint.sh && \
    /usr/local/bin/install_bonsai.sh "${PYTORCH_OPTION}"

ENTRYPOINT ["/usr/local/bin/container_entrypoint.sh"]
CMD ["bash"]
