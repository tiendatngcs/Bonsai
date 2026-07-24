#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn resnet152 256 13.99 rockmate rockmate_resnet152_batch256_large_sched.pkl "$@"
