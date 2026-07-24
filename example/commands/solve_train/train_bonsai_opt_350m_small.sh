#!/usr/bin/env bash
set -euo pipefail
python -m example.train_transformer --model facebook/opt-350m --batch-size 8 --budget-gb 8.0 --model-weights-mb 1263.41 --scheduler bonsai "$@"
