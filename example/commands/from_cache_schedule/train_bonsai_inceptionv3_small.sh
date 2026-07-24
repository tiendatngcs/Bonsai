#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn inceptionv3 256 5.32 bonsai bonsai_inceptionv3_batch256_small_sched.pkl "$@"
