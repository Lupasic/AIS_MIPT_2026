#!/usr/bin/env bash
set -euo pipefail

WORKERS="${1:-2}"

echo "[ray] starting head + ${WORKERS} workers"
docker compose -f docker/docker-compose.distributed.yml up -d --build ray-head --scale ray-worker="${WORKERS}" ray-worker

echo "[ray] waiting for cluster"
sleep 5

echo "[ray] running tune demo"
docker compose -f docker/docker-compose.distributed.yml exec ray-head python scripts/ray_tune_cpu_demo.py

echo "[ray] stop with: docker compose -f docker/docker-compose.distributed.yml down"
