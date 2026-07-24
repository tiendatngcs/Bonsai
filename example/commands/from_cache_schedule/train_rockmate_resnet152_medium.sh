#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn resnet152 256 10.49 rockmate rockmate_resnet152_batch256_medium_sched.pkl "$@"
