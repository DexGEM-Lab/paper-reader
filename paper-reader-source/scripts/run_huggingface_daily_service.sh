#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPOSITORY_DIR=$(cd -- "${PROJECT_DIR}/.." && pwd)

cd "${REPOSITORY_DIR}"
export PYTHONPATH="${REPOSITORY_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"

exec python3 -u -m paper_reader_source.service
