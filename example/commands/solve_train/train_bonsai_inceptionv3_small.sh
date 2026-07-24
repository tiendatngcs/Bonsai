#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model inceptionv3 --batch-size 256 --budget-gb 5.8 --model-weights-mb 85.14 --scheduler bonsai "$@"
