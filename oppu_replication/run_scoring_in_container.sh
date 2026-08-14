#!/bin/bash
# Run a scoring/plotting entrypoint inside the ver4 container, which lacks the
# analysis-layer deps (evaluate, scipy — scoring previously ran in .venv-audit
# on a login node, now banned by the 2 GB login cgroup rule). Installs them
# into the job's ephemeral container, then execs the given python script+args.
set -euo pipefail
cd "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip install --quiet --no-input evaluate scipy absl-py nltk
exec python3 "$@"
