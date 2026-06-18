#!/bin/bash

cd "$(dirname "$0")" || exit

# NOTE: This deployment uses a hand-managed uv environment: a custom ROCm nightly
# torch built for this AMD GPU (gfx1151) plus an editable, local exllamav3. The
# stock start.py auto-runs `uv pip install .[amd]` on first launch, which would
# clobber that torch stack and replace the local exllamav3 with a release wheel.
# So we activate the existing venv and launch main.py directly, leaving all
# dependency management to the manual uv setup.

if [ -n "$CONDA_PREFIX" ]; then
    echo "It looks like you're in a conda environment. Skipping venv check."
elif [ -d "venv" ]; then
    echo "Activating venv"
    # shellcheck source=/dev/null
    source venv/bin/activate
else
    echo "ERROR: venv not found."
    echo "This deployment relies on a manually-built ROCm environment rather than"
    echo "start.py's auto-installer (which would install an incompatible torch and"
    echo "overwrite the local exllamav3). Create the environment first, e.g.:"
    echo "  uv venv venv -p 3.12"
    echo "  uv pip install --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch triton"
    echo "  uv pip install -e ."
    echo "  EXLLAMA_NOCOMPILE=1 uv pip install -e ../exllamav3 --no-deps --no-build-isolation"
    exit 1
fi

# Create a default config on first run (mirrors the old start.py behaviour)
if [ ! -f "config.yml" ]; then
    echo "config.yml not found; creating one from config_sample.yml"
    cp config_sample.yml config.yml
fi

python3 main.py "$@"
