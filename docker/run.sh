#!/usr/bin/env bash
# Run any weight_space_bench command inside the GPU image, e.g.  docker/run.sh run all
# The benchmark folder is mounted at the same absolute path as on the host, so the kernel containers the driver
# starts (through the host's Docker socket) can mount the same folders.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
exec docker run --rm -it --gpus all --ipc=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HERE":"$HERE" -w "$HERE" \
  -e OPENROUTER_API_KEY -e HF_TOKEN -e WSB_DATA -e WSB_RESULTS \
  -e HF_HOME="$HERE/data/hf_cache" \
  --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /var/run/docker.sock)" \
  weight-space-bench-gpu:latest python -m weight_space_bench math-edit "$@"
