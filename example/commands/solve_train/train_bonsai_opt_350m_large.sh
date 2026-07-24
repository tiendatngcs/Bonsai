#!/usr/bin/env bash
set -euo pipefail
python -m example.train_transformer --model facebook/opt-350m --batch-size 8 --budget-gb 14.3 --model-weights-mb 1263.41 --scheduler bonsai "$@"
