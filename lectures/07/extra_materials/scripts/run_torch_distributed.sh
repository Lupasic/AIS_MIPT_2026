#!/usr/bin/env bash
set -euo pipefail

WORLD_SIZE="${1:-2}"

echo "[torch.distributed] starting ${WORLD_SIZE} CPU nodes"
WORLD_SIZE="${WORLD_SIZE}" docker compose -f docker/docker-compose.distributed.yml up --build --scale torch-node="${WORLD_SIZE}" torch-node
