#!/usr/bin/env bash
set -euo pipefail
python -m example.train_transformer --model EleutherAI/pythia-160m --batch-size 16 --budget-gb 7.0 --scheduler rockmate "$@"
