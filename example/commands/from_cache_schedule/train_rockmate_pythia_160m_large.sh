#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer EleutherAI/pythia-160m 16 17.33 rockmate rockmate_EleutherAI_pythia-160m_batch16_large_sched.pkl "$@"
