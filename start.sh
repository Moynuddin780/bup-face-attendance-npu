#!/bin/bash
# Kept for muscle memory: the old launcher. Everything now goes through run.sh
# (./setup.sh once, then ./run.sh — or Docker: docker compose up -d).
exec "$(cd "$(dirname "$0")" && pwd -P)/run.sh" "$@"
