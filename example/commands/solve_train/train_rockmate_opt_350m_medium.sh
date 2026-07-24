#!/usr/bin/env bash
set -euo pipefail
python -m example.train_transformer --model facebook/opt-350m --batch-size 8 --budget-gb 10.8 --scheduler rockmate "$@"
