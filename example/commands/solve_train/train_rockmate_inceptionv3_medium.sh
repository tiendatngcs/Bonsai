#!/usr/bin/env bash
set -euo pipefail
python -m example.train_cnn --model inceptionv3 --batch-size 256 --budget-gb 8.2 --scheduler rockmate "$@"
