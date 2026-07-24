#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model resnet152 --batch-size 256 --budget-gb 13.3 --scheduler rockmate "$@"
