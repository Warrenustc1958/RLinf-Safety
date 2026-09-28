#!/bin/bash
set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${REPO_PATH}/evaluations/run_safelibero.sh" "$@"
