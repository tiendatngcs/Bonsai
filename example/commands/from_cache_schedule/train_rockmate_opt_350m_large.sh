#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer facebook/opt-350m 8 20.02 rockmate rockmate_facebook_opt-350m_batch8_large_sched.pkl "$@"
