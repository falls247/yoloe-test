#!/usr/bin/env bash
set -euo pipefail
uv sync
exec uv run python app.py
