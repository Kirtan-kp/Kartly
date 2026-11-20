#!/usr/bin/env bash
# Build the image, start the whole stack with throwaway secrets, check it works and is locked down,
# then remove it. Needs Docker. Used by CI; also fine to run by hand:  scripts/compose-smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="$(mktemp)"
PORT="${SMOKE_PORT:-18000}"
KEY="smoke$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
cat > "$ENV_FILE" <<ENV
POSTGRES_PASSWORD=smoke$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')
KARTLY_API_KEY=$KEY
KARTLY_PORT=$PORT
ENV
# The compose file names .env.deploy as its env_file; point it at the throwaway file instead.
cp "$ENV_FILE" .env.deploy.smoke
sed 's#\.env\.deploy#.env.deploy.smoke#' docker-compose.deploy.yml > docker-compose.smoke.yml
dc() { docker compose --env-file "$ENV_FILE" -f docker-compose.smoke.yml "$@"; }

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- smoke test failed; container state and logs follow"
    dc ps -a || true
    dc logs --no-color --tail 80 || true
  fi
  dc down -v --remove-orphans >/dev/null 2>&1 || true
  rm -f "$ENV_FILE" .env.deploy.smoke docker-compose.smoke.yml
  exit "$status"
}
trap cleanup EXIT

step() { echo "==> $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

step "build and start"
dc up -d --build

step "wait for the API to become healthy (up to 3 minutes)"
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/healthz" | grep -q '"ok"' || fail "/healthz did not report ok"

step "the migration ran to completion and the database is at the latest revision"
[ "$(docker inspect -f '{{.State.ExitCode}}' "$(dc ps -aq migrate)")" = "0" ] || fail "migrate did not exit 0"
dc exec -T api python -m kartly.migrate --check || fail "database is not at head"

step "the admin page is served, and the data routes refuse an unauthenticated caller"
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/admin")" = "200" ] || fail "/admin"
curl -sD - -o /dev/null "http://127.0.0.1:$PORT/admin" | grep -qi '^content-security-policy:' || fail "no CSP header"
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/tickets")" = "401" ] || fail "tickets open"
[ "$(curl -s -o /dev/null -w '%{http_code}' -H "X-API-Key: $KEY" "http://127.0.0.1:$PORT/v1/metrics")" = "200" ] || fail "key rejected"

step "the containers are locked down"
[ "$(dc exec -T api id -u)" = "10001" ] || fail "the API runs as root"
if dc exec -T api sh -c 'touch /app/should-not-work' 2>/dev/null; then fail "the filesystem is writable"; fi
if dc port postgres 5432 2>/dev/null | grep -q .; then fail "postgres is published on the host"; fi

step "the worker stops cleanly when told to"
dc stop worker
[ "$(docker inspect -f '{{.State.ExitCode}}' "$(dc ps -aq worker)")" = "0" ] || fail "the worker did not exit 0"

echo "smoke test passed"
