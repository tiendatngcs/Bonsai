#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer EleutherAI/pythia-160m 16 11.03 bonsai bonsai_EleutherAI_pythia-160m_batch16_small_sched.pkl "$@"
