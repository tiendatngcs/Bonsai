#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn googlenet 256 2.4 rockmate rockmate_googlenet_batch256_medium_sched.pkl "$@"
