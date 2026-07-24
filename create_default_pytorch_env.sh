
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# first argument is the option
# option="bonsai_exec_only"
option="bonsai"
env_name="pytorch_env_$option"

# ensure that jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found, please install jq to proceed."
    echo "On Ubuntu, you can install it via: sudo apt-get install jq"
    exit 1
fi

# read values from pytorch_compatibility.json based on the option
PYTORCH_TYPE=$(jq -r ".\"$option\".pytorch_type" "$SCRIPT_DIR/pytorch_compatibility.json")
PYTORCH_REPO=$(jq -r ".\"$option\".pytorch_repo" "$SCRIPT_DIR/pytorch_compatibility.json")
PYTORCH_VERSION=$(jq -r ".\"$option\".pytorch_version" "$SCRIPT_DIR/pytorch_compatibility.json")
TORCHVISION_VERSION=$(jq -r ".\"$option\".torchvision_version" "$SCRIPT_DIR/pytorch_compatibility.json")
PYTHON_VERSION=$(jq -r ".\"$option\".python_version" "$SCRIPT_DIR/pytorch_compatibility.json")
TRANSFORMERS_VERSION=$(jq -r ".\"$option\".transformers_version" "$SCRIPT_DIR/pytorch_compatibility.json")
PATCH_FILE=$(jq -r ".\"$option\".patch_file" "$SCRIPT_DIR/pytorch_compatibility.json")

if [ "$PATCH_FILE" != "null" ]; then
    PATCH_FILE="$SCRIPT_DIR/$PATCH_FILE"
fi

if [ "$PYTORCH_VERSION" == "null" ] || [ "$TORCHVISION_VERSION" == "null" ] || [ "$PYTHON_VERSION" == "null" ] || [ "$TRANSFORMERS_VERSION" == "null" ]; then
    echo "Invalid option provided. Please choose a valid option."
    exit 1
fi


echo "Creating conda environment with PyTorch $PYTORCH_VERSION, TorchVision $TORCHVISION_VERSION, Python $PYTHON_VERSION, Transformers $TRANSFORMERS_VERSION"

conda create -n $env_name python=$PYTHON_VERSION -y
conda activate $env_name
conda install pip -y

# if pytorch type is from_source, install from pytorch repos
if [ "$PYTORCH_TYPE" == "from_source" ]; then
    # PYTORCH_REPO must not be null
    # if [ "$PYTORCH_REPO" == "null" ]; then
    #     echo "Pytorch repo URL is missing for from_source option."
    #     # exit 1
    # fi
    echo Installing PyTorch from $PYTORCH_REPO

    conda install -c conda-forge "gcc=12" "gxx=12"

    # Checkout tag based on version

    source "$SCRIPT_DIR/install_pytorch_from_source.sh" "$PYTORCH_REPO" "$PYTORCH_VERSION" "$PATCH_FILE"

    source "$SCRIPT_DIR/install_torchvision_from_source.sh" "$TORCHVISION_VERSION"

    pip install transformers==$TRANSFORMERS_VERSION
elif [ "$PYTORCH_TYPE" == "pip_prebuilt" ]; then
    echo "Installing prebuilt PyTorch binaries"
    # install prebuilt pytorch via pip
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION

    pip install transformers==$TRANSFORMERS_VERSION
else
    echo "Invalid pytorch type specified in the configuration."
    # exit 1
fi

# pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION transformers==$TRANSFORMERS_VERSION
# pip install transformers==$TRANSFORMERS_VERSION

echo "Conda environment '$env_name' created and activated with PyTorch $PYTORCH_VERSION, TorchVision $TORCHVISION_VERSION, Python $PYTHON_VERSION"

# run this bash script with
# source create_default_pytorch_env.sh <option>