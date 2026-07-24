gpu_name=RTX6000
# if [ -z "$gpu_name" ]; then
#     echo "Usage: $0 <gpu_name>"
#     echo "Example: $0 RTX6000"
#     exit 1
# fi

# ensure that zip and unzip are installed
if ! command -v zip &> /dev/null || ! command -v unzip &> /dev/null; then
    echo "zip and/or unzip could not be found, please install them to proceed."
    echo "On Ubuntu, you can install them via: sudo apt-get install zip unzip"
    exit 1
fi

MODELS=("opt-350m" "googlenet" "inceptionv3" "gpt2" "resnet18" "resnet50" "resnet152" "pythia-160m")
MEMORY_MODES=("rockmate" "checkmate" "bonsai")

# For each model in MODELS, zip the corresponding pkl files
for model in "${MODELS[@]}"; do
    for memory_mode in "${MEMORY_MODES[@]}"; do
        echo "Unzipping schedules for model: $model with memory mode: $memory_mode"
        unzip pkl_schedules_${gpu_name}_${memory_mode}_${model}.zip
    done
done

