#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn googlenet 256 1.6 bonsai bonsai_googlenet_batch256_small_sched.pkl "$@"
