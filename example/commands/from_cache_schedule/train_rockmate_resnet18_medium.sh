#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn resnet18 256 1.13 rockmate rockmate_resnet18_batch256_medium_sched.pkl "$@"
