#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer openai-community/gpt2 8 8.66 rockmate rockmate_openai-community_gpt2_batch8_small_sched.pkl "$@"
