#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer EleutherAI/pythia-160m 16 13.83 rockmate rockmate_EleutherAI_pythia-160m_batch16_medium_sched.pkl "$@"
