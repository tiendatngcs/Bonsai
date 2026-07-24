#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer openai-community/gpt2 8 10.3 rockmate rockmate_openai-community_gpt2_batch8_medium_sched.pkl "$@"
