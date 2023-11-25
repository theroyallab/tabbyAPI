#!/bin/bash
cd "$(dirname "$0")"
source ./venv/bin/activate
pip -V
python main.py
