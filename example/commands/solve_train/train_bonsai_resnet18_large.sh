#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model resnet18 --batch-size 256 --budget-gb 1.33 --model-weights-mb 42.8 --scheduler bonsai "$@"
