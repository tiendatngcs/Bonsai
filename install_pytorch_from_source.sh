# pip install -r requirements.txt

GITHUB_REPO=$1
PYTORCH_VERSION=$2
PATCH_FILE=$3


# function to extrace repo name from repo url, return the repo name
# for example git@github.com:tiendatngcs/pytorch.git --> pytorch
extract_repo_name() {
    local repo_url=$1
    local repo_name_with_git=${repo_url##*/}
    local repo_name=${repo_name_with_git%.git}
    echo $repo_name
}

# GIT_REPO must not be empty or empty string
if [ -z "$GITHUB_REPO" ]; then
    echo "GitHub repository URL is missing. Please provide a valid repository URL."
    exit 1
fi

repo_name=$(extract_repo_name $GITHUB_REPO)

# operates at $HOME
pushd $HOME > /dev/null

git clone --branch "v${PYTORCH_VERSION}" --depth 1 --recursive "${GITHUB_REPO}" "${repo_name}"
# git clone --recursive $GITHUB_REPO


pushd ./$repo_name > /dev/null
git submodule update --init --recursive
# git checkout tags/v$PYTORCH_VERSION

# apply patch if patch file is not null
if [ "$PATCH_FILE" != "null" ]; then
    if [ ! -f "$PATCH_FILE" ]; then
        echo "Patch file '$PATCH_FILE' does not exist."
        return 1
    fi

    echo "Applying patch $PATCH_FILE"
    if ! git apply "$PATCH_FILE"; then
        echo "Failed to apply patch '$PATCH_FILE'."
        return 1
    fi
fi

git submodule sync
git submodule update --init --recursive

pip install pyyaml
pip install typing_extensions
pip install tensorboard
pip install tqdm

# PyTorch 2.3's vendored protobuf does not configure cleanly with CMake 4.x.
conda install -c conda-forge "cmake=3.31.1" ninja -y
# export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}" # this is already done in install_anaconda.sh

conda install -c conda-forge libstdcxx-ng -y

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
export PYTORCH_HOME="$HOME/$repo_name"


# python setup.py install

export USE_NUMPY=1

pip install --no-build-isolation -e .

popd
popd
