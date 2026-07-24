#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GUROBI_HOME:-}" ]]; then
    export PATH="${GUROBI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

exec "$@"

