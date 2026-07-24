#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn resnet152 256 7.69 bonsai bonsai_resnet152_batch256_small_sched.pkl "$@"
