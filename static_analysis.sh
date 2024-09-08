#!/bin/bash

if command -v less > /dev/null 2>&1; then
    mypy start.py | less
else
    mypy start.py
fi