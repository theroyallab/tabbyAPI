#!/usr/bin/env python3
"""Bump pinned exllamav3 wheel URLs in pyproject.toml to a newer release.

Usage (run from the repo root):
    python tools/update_exllamav3.py [--version v1.5.4] [--file pyproject.toml] [--dry-run]
"""

import argparse
import difflib
import json
import re
import sys
import urllib.error
from pathlib import Path
from urllib.request import Request, urlopen

GITHUB_API_V3 = "https://api.github.com/repos/turboderp-org/exllamav3/releases/latest"
VERSION_RE = re.compile(r"^v?\d+\.\d+\.\d+.*$")


def fetch_latest_version() -> str:
    """Return the latest exllamav3 release tag (e.g. 'v1.5.3')."""
    req = Request(
        GITHUB_API_V3,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "tabbyapi-update-script",
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not fetch latest exllamav3 release: {exc}") from exc
    tag = data.get("tag_name") if isinstance(data, dict) else None
    if not tag:
        raise RuntimeError("Could not determine latest exllamav3 version from GitHub API.")
    return tag


def normalize_tag(tag: str) -> str:
    """Validate a version tag and ensure it starts with 'v'."""
    tag = tag.strip()
    if not VERSION_RE.match(tag):
        raise ValueError(f"Invalid version tag: {tag!r} (expected e.g. v1.5.4).")
    return tag if tag.startswith("v") else f"v{tag}"


def bump_exllamav3_urls(text: str, new_tag: str) -> tuple[str, int, list[str]]:
    """
    Replace exllamav3 wheel URLs in *text* with *new_tag*.

    Returns (new_text, replacements_made, warnings).
    """
    new_ver = new_tag.removeprefix("v")
    warnings: list[str] = []

    # Detect every distinct version currently pinned in exllamav3 URLs.
    found_versions = set(
        re.findall(
            r"https://github\.com/turboderp-org/exllamav3/releases/download/v([^/]+)/exllamav3-",
            text,
        )
    )
    if not found_versions:
        return text, 0, ["Could not detect current exllamav3 version in pyproject.toml."]
    if len(found_versions) > 1:
        warnings.append(
            "Multiple exllamav3 versions pinned "
            f"({', '.join(sorted(found_versions))}); bumping all to {new_tag}."
        )
    if found_versions == {new_ver}:
        return text, 0, []

    total_lines = len(
        re.findall(
            r"exllamav3 @ https://github\.com/turboderp-org/exllamav3/releases/download/",
            text,
        )
    )

    count = 0
    new_text = text
    for old_ver in sorted(found_versions):
        if old_ver == new_ver:
            continue
        # Replace the version in both the download path and the wheel filename.
        # We match: download/vOLD/exllamav3-OLD[+ or %2B or -]
        pattern = re.compile(
            rf"(https://github\.com/turboderp-org/exllamav3/releases/download/)"
            rf"v{re.escape(old_ver)}/"
            rf"(exllamav3-)"
            rf"{re.escape(old_ver)}"
            rf"(\+|%2B|-)"  # the next char after the version in the wheel name
        )

        def repl(match: re.Match) -> str:
            nonlocal count
            count += 1
            return f"{match.group(1)}v{new_ver}/{match.group(2)}{new_ver}{match.group(3)}"

        new_text = pattern.sub(repl, new_text)

    # Catch a partial bump: old download paths left behind.
    leftover = re.findall(
        r"https://github\.com/turboderp-org/exllamav3/releases/download/v([^/]+)/exllamav3-",
        new_text,
    )
    leftover_old = sorted({v for v in leftover if v != new_ver})
    if leftover_old:
        warnings.append(
            f"Partial bump: these versions remain after rewriting: {', '.join(leftover_old)}."
        )
    if count != total_lines and total_lines:
        warnings.append(
            f"Rewrote {count} of {total_lines} exllamav3 URL(s); check for unexpected URL formats."
        )

    return new_text, count, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bump pinned exllamav3 wheel URLs in pyproject.toml to a newer release."
    )
    parser.add_argument(
        "--version",
        metavar="TAG",
        help="Target version tag (e.g. v1.5.4). Defaults to the latest GitHub release.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml (default: ./pyproject.toml, run from repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the diff without modifying pyproject.toml.",
    )
    args = parser.parse_args()

    pyproject: Path = args.file
    if not pyproject.exists():
        print(f"ERROR: {pyproject} not found.", file=sys.stderr)
        return 1

    original_text = pyproject.read_text(encoding="utf-8")

    if args.version:
        try:
            new_tag = normalize_tag(args.version)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Using requested version: {new_tag}")
    else:
        try:
            new_tag = fetch_latest_version()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Latest exllamav3 release: {new_tag}")

    new_text, count, warnings = bump_exllamav3_urls(original_text, new_tag)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    if count == 0:
        if not warnings:
            print("pyproject.toml is already up to date.")
        return 0 if not warnings else 1

    print(f"Updated {count} exllamav3 reference(s).")

    if args.dry_run:
        diff = difflib.unified_diff(
            original_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(pyproject),
            tofile=str(pyproject) + " (updated)",
        )
        sys.stdout.writelines(diff)
        return 0

    pyproject.write_text(new_text, encoding="utf-8")
    print(f"Wrote changes to {pyproject}.")
    print("Run `python start.py --update-deps` to reinstall with the new wheels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
