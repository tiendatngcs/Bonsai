#!/usr/bin/env bash
set -euo pipefail

require_docker_access() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is not installed or not on PATH." >&2
        exit 1
    fi

    local docker_info_output
    if ! docker_info_output="$(docker info 2>&1)"; then
        if grep -qi "permission denied" <<<"${docker_info_output}"; then
            cat >&2 <<EOF
Docker is installed, but the current user cannot access the Docker daemon.
Fix one of these and try again:
  - run the script through sudo
  - add your user to the docker group, then start a new shell/session

Docker reported:
${docker_info_output}
EOF
        elif grep -Eqi "Cannot connect to the Docker daemon|is the docker daemon running" <<<"${docker_info_output}"; then
            cat >&2 <<EOF
Docker daemon is not reachable. Start Docker and try again.

Docker reported:
${docker_info_output}
EOF
        else
            cat >&2 <<EOF
Docker is installed, but the daemon is not reachable from this shell.

Docker reported:
${docker_info_output}
EOF
        fi
        exit 1
    fi
}
