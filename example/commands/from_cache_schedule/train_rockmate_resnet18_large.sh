#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn resnet18 256 1.48 rockmate rockmate_resnet18_batch256_large_sched.pkl "$@"
