VERSION=$1


# VERSION must be provided
if [ -z "$VERSION" ]; then
    echo "Version argument is missing. Please provide a version (e.g., 0.18.0)."
    exit 1
fi

# Operates at $HOME
pushd $HOME > /dev/null

git clone --branch "v${VERSION}" --depth 1 https://github.com/pytorch/vision.git vision
pushd ./vision > /dev/null

export LD_LIBRARY_PATH=$PYTORCH_HOME/build/lib:$LD_LIBRARY_PATH

pip install "setuptools<81"
pip install --no-build-isolation -e .

popd
popd
