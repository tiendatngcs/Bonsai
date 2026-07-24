#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/run_cached_training.sh" transformer EleutherAI/pythia-160m 16 13.83 bonsai bonsai_EleutherAI_pythia-160m_batch16_medium_sched.pkl "$@"
