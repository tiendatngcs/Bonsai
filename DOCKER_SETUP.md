# Docker setup for Bonsai

This repository currently documents a Conda-based installation. The files added here provide a Docker-based alternative that now mirrors the same source-build behavior:

1. create a Python 3.11 environment on top of a CUDA 12.1 devel image
2. build the patched PyTorch 2.3.0 from source with CUDA enabled, or install CUDA-enabled prebuilt wheels
3. build TorchVision 0.18.0 from source against that PyTorch build when needed
4. install `rockmate-bonsai` in editable mode

The Docker workflow is intended for Linux hosts with Docker installed.

## What gets added

- `Dockerfile` — builds a reusable Bonsai image
- `docker/install_bonsai.sh` — install Bonsai, including tracer and execution engine.
- `docker/build_image.sh` — convenience wrapper around `docker build`
- `docker/run_container.sh` — runs the image with the repo mounted and GPU access

## Prerequisites

- Docker installed and usable from your shell (`docker info` should succeed without `permission denied`)
- enough disk and memory for building PyTorch from source
- NVIDIA Container Toolkit if you want GPU access inside the container
- an NVIDIA driver on the host that is compatible with CUDA 12.1 containers

## Build the image

From the repository root:

```bash
./docker/build_image.sh
```

The script performs a Docker preflight check first and now distinguishes between:

- Docker not running
- Docker socket permission problems
- other daemon connectivity errors

By default this builds:

- the `bonsai` compatibility profile from `pytorch_compatibility.json`
- the image tag `bonsai:dev`

### What `bonsai` means

The default `bonsai` profile is the expensive path:

- it clones PyTorch source
- applies `bonsai.patch`
- compiles patched PyTorch inside Docker
- compiles TorchVision from source against that build

If your goal is only to verify the Docker flow first, use the prebuilt profile instead:

```bash
PYTORCH_OPTION=bonsai_exec_only ./docker/build_image.sh
```

That image is much faster to build, but it does **not** include the customized PyTorch tracer build used by Bonsai.
It now installs CUDA-enabled PyTorch/TorchVision wheels from the official `cu121` wheel index.

You can override both:

```bash
IMAGE_NAME=bonsai:custom PYTORCH_OPTION=bonsai ./docker/build_image.sh
```

You can also reduce or increase PyTorch parallelism during the source build:

```bash
MAX_JOBS=2 ./docker/build_image.sh
```

The CUDA source build now defaults to:

```bash
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
```

That avoids PyTorch 2.3's older CMake logic tripping over newer names such as `9.0a`.
If you need a different set of targets, override it when building:

```bash
TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" ./docker/build_image.sh
```

## Common build failures

The current Docker installer tracks the same fixes as the Conda scripts:

- pinned `cmake==3.31.1` so PyTorch 2.3's vendored protobuf does not trip over CMake 4.x
- `pip install --no-build-isolation -e .` for the PyTorch and TorchVision source installs so editable builds reuse the environment that already has `cmake` and `ninja`
- `PYTORCH_HOME` exported from the PyTorch build before TorchVision is installed
- Docker now builds from a CUDA 12.1 devel base image instead of a CPU-only Python image

### Docker daemon not running

Symptom:

```text
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

Fix:

- start Docker
- rerun `./docker/build_image.sh`

### Docker socket permission denied

Symptom:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

Fix:

- run `docker info` directly to confirm the same error
- either run the script with `sudo`, or add your user to the `docker` group
- after changing group membership, start a new shell/session before rerunning the script

### Unknown `PYTORCH_OPTION`

Symptom:

```text
Unknown PYTORCH_OPTION '...'
```

Fix:

- use one of the keys in `pytorch_compatibility.json`
- currently that means `bonsai_exec_only` or `bonsai`

### Patched PyTorch source build fails or is killed

This is the most likely failure mode for the default `bonsai` profile.

Typical causes:

- not enough RAM
- not enough CPU or disk
- a long compile being mistaken for a hang

Things to try:

```bash
DOCKER_PROGRESS=plain MAX_JOBS=2 ./docker/build_image.sh
```

or first validate the Docker flow with:

```bash
PYTORCH_OPTION=bonsai_exec_only ./docker/build_image.sh
```

`DOCKER_PROGRESS=plain` is useful because it shows the underlying build step that failed.

### Older image build hits `ninja` or CMake compatibility errors

Symptoms from an outdated Docker install script usually look like one of:

```text
Preparing editable metadata (pyproject.toml) did not run successfully
...
Running '.../ninja' '--version' failed with: no such file or directory
```

or:

```text
CMake Error at third_party/protobuf/cmake/CMakeLists.txt:2 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

Fix:

- rebuild the image from the current repository state with `./docker/build_image.sh`
- if needed, force a clean rebuild with `docker build --no-cache ...` via the same script inputs

### CUDA source build fails with `Unknown CUDA Architecture Name 9.0a`

PyTorch 2.3's CUDA CMake helpers do not recognize the Hopper-specific `9.0a` architecture string.

Fix:

- rebuild with the current scripts, which pin `TORCH_CUDA_ARCH_LIST` to a compatible list ending in `9.0`
- if you need to narrow targets, pass a custom list such as:

```bash
TORCH_CUDA_ARCH_LIST="8.9;9.0" ./docker/build_image.sh
```

## Run the container

The standard development shell is:

```bash
./docker/run_container.sh
```

This mounts the repository at `/workspace/Bonsai`, so edits on the host are visible in the container.

### Disable GPU access

The container uses all host GPUs by default. To run without GPU access:

```bash
USE_GPU=0 ./docker/run_container.sh
```

GPU execution requires an NVIDIA Container Toolkit-enabled Docker host. By default,
the script adds `--gpus all` and the standard NVIDIA runtime environment variables
so PyTorch can see the host GPUs.

## Verify the environment inside the container

Once inside the container:

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import rockmate; print('rockmate import ok')"
```

## Example usage in Docker

The existing examples run the same way:

```bash
python example_tracer.py
python example_train.py
```

The repository mount means outputs such as `./traces` and `./data` persist on the host.

## Rebuild when setup inputs change

Rebuild the image when any of these change:

- `bonsai.patch`
- `pytorch_compatibility.json`
- `docker/install_bonsai.sh`
- Docker base image or OS-level dependencies

```bash
./docker/build_image.sh
```
