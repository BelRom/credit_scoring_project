#!/usr/bin/env bash
set -e

docker compose up -d
echo "API:      http://localhost:8000/docs"
echo "Frontend: http://localhost:8501"
