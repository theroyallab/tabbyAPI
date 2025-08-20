#!/bin/bash

cd "$(dirname "$0")" || exit

if command -v uv >/dev/null 2>&1; then
    HAS_UV=1
else
    HAS_UV=0
fi

if [ -n "$CONDA_PREFIX" ]; then
    echo "It looks like you're in a conda environment. Skipping venv check."
else
    if [ ! -d "venv" ]; then
        echo "Venv doesn't exist! Creating one for you."

        if [ "$HAS_UV" -eq 1 ]; then
            echo "It looks like you're using uv. Running appropriate commands."
            uv venv venv -p 3.12
        else
            python3 -m venv venv
        fi

        if [ -f "start_options.json" ]; then
            echo "Removing old start_options.json"
            rm -rf start_options.json
        fi
    fi

    echo "Activating venv"

    # shellcheck source=/dev/null
    source venv/bin/activate
fi

python3 start.py "$@"
