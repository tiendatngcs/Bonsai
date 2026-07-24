#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer facebook/opt-350m 8 13.72 bonsai bonsai_facebook_opt-350m_batch8_small_sched.pkl "$@"
