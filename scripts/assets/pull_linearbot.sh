#!/usr/bin/env bash
# Download linearbot assets from HuggingFace into the MlSpaces assets directory.
# Usage: bash scripts/assets/pull_linearbot.sh
set -euo pipefail

DEST=${MLSPACES_ASSETS_DIR:-$HOME/robot_assets}

hf download TreeePlanter/linearbot-assets \
    --repo-type dataset \
    --include "robots/linearbot/**" \
    --local-dir "$DEST"

echo "Assets at: $DEST/robots/linearbot"
