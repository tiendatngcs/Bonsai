#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
    echo "Usage: $0 <cnn|transformer> <model> <batch-size> <budget-gb> <bonsai|rockmate> <schedule-file> [training arguments...]" >&2
    exit 1
fi

model_type=$1
model=$2
batch_size=$3
budget_gb=$4
scheduler=$5
schedule_file=$6
shift 6

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${root_dir}"

case "${model_type}" in
    cnn)
        command=(python -m example.train_cnn --model "${model}" --batch-size "${batch_size}")
        ;;
    transformer)
        command=(python -m example.train_transformer --model "${model}" --batch-size "${batch_size}")
        ;;
    *)
        echo "Unsupported model type '${model_type}'." >&2
        exit 1
        ;;
esac

if [[ "${scheduler}" == "bonsai" ]]; then
    case "${model}" in
        resnet18) model_weights_mb=42.8 ;;
        resnet50) model_weights_mb=90.43 ;;
        googlenet) model_weights_mb=24.42 ;;
        inceptionv3) model_weights_mb=85.14 ;;
        resnet152) model_weights_mb=222.55 ;;
        EleutherAI/pythia-160m) model_weights_mb=619.21 ;;
        openai-community/gpt2) model_weights_mb=474.7 ;;
        facebook/opt-350m) model_weights_mb=1263.41 ;;
        *)
            echo "No model weight is configured for Bonsai model '${model}'." >&2
            exit 1
            ;;
    esac
    command+=(--model-weights-mb "${model_weights_mb}")
fi

exec "${command[@]}" \
    --budget-gb "${budget_gb}" \
    --scheduler "${scheduler}" \
    --schedule-file "./schedules/${schedule_file}" \
    "$@"
