#!/usr/bin/env bash
set -euo pipefail
python -m example.train_transformer --model openai-community/gpt2 --batch-size 8 --budget-gb 6.0 --model-weights-mb 474.7 --scheduler bonsai "$@"
