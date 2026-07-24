#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" cnn googlenet 256 3.4 bonsai bonsai_googlenet_batch256_large_sched.pkl "$@"
