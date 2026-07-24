#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model resnet50 --batch-size 256 --budget-gb 6.6 --scheduler rockmate "$@"
