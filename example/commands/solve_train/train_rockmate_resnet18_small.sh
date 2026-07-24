#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model resnet18 --batch-size 256 --budget-gb 0.7 --scheduler rockmate "$@"
