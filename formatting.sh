#!/usr/bin/env bash
# YAPF formatter, adapted from ray and skypilot.
#
# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash formatting.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

RUFF_VERSION=$(ruff --version | head -n 1 | awk '{print $2}')


# Format and lint all files
format_and_lint() {
    ruff format
    ruff check
}

# Call format command
format_and_lint
echo 'tabbyAPI ruff lint and format: Done'

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi
