#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn resnet18 256 0.85 rockmate rockmate_resnet18_batch256_small_sched.pkl "$@"
