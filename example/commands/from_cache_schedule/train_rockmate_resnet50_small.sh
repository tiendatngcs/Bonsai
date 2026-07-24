#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn resnet50 256 3.3 rockmate rockmate_resnet50_batch256_small_sched.pkl "$@"
