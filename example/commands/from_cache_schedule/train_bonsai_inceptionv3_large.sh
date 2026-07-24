#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn inceptionv3 256 10.92 bonsai bonsai_inceptionv3_batch256_large_sched.pkl "$@"
