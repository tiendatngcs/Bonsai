#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model googlenet --batch-size 256 --budget-gb 3.3 --model-weights-mb 24.42 --scheduler bonsai "$@"
