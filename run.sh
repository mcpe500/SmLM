#!/bin/bash
# Run a Python script using the conda environment
export PATH="/root/.local/share/mamba/envs/slm/bin:$PATH"
exec python "$@"
