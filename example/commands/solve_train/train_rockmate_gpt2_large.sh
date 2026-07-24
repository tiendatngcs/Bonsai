#!/usr/bin/env bash
set -euo pipefail
python -m example.train_transformer --model openai-community/gpt2 --batch-size 8 --budget-gb 9.69 --scheduler rockmate "$@"
