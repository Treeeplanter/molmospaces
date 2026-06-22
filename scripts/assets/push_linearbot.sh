#!/usr/bin/env bash
# Push linearbot assets to HuggingFace.
# Usage: bash scripts/assets/push_linearbot.sh
set -euo pipefail

ROBOT_DIR=${ROBOT_DIR:-/home/shuo/research/resources/mlspaces_assets/robots/linearbot}

hf upload TreeePlanter/linearbot-assets "$ROBOT_DIR" robots/linearbot \
    --repo-type dataset
