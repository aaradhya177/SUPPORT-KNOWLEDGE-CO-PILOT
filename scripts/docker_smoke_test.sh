#!/usr/bin/env bash
set -euo pipefail

COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-support-knowledge-copilot-smoke}"
export COMPOSE_PROJECT_NAME

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SUPPORT_COPILOT_API_KEY="${SUPPORT_COPILOT_API_KEY:-}"
SUPPORT_COPILOT_ADMIN_API_KEY="${SUPPORT_COPILOT_ADMIN_API_KEY:-}"

if [[ -z "${SUPPORT_COPILOT_API_KEY}" || -z "${SUPPORT_COPILOT_ADMIN_API_KEY}" ]]; then
  echo "Set SUPPORT_COPILOT_API_KEY and SUPPORT_COPILOT_ADMIN_API_KEY before running the smoke test."
  exit 1
fi

cleanup() {
  docker compose down --volumes --remove-orphans
}
trap cleanup EXIT

docker compose up --build -d

echo "Waiting for backend healthcheck..."
for _ in {1..60}; do
  status="$(docker inspect --format='{{.State.Health.Status}}' "${COMPOSE_PROJECT_NAME}-backend-1" 2>/dev/null || true)"
  if [[ "${status}" == "healthy" ]]; then
    break
  fi
  sleep 2
done

curl -fsS http://localhost:8000/health

echo
echo "Starting ingestion..."
curl -fsS -X POST \
  -H "X-API-Key: ${SUPPORT_COPILOT_ADMIN_API_KEY}" \
  http://localhost:8000/api/v1/ingest

echo
echo "Waiting for ingestion to finish..."
for _ in {1..120}; do
  ingest_status="$(
    curl -fsS \
      -H "X-API-Key: ${SUPPORT_COPILOT_ADMIN_API_KEY}" \
      http://localhost:8000/api/v1/ingest/status |
      python -c "import json,sys; print(json.load(sys.stdin)['status'])"
  )"
  if [[ "${ingest_status}" == "completed" ]]; then
    break
  fi
  if [[ "${ingest_status}" == "failed" ]]; then
    echo "Ingestion failed"
    exit 1
  fi
  sleep 2
done

echo "Querying backend..."
curl -fsS \
  -H "X-API-Key: ${SUPPORT_COPILOT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query":"What should I check if a password reset email never arrives?","top_k":3}' \
  http://localhost:8000/api/v1/query

echo
echo "Docker smoke test completed."
