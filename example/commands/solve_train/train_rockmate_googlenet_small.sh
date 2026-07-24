#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model googlenet --batch-size 256 --budget-gb 1.5 --scheduler rockmate "$@"
