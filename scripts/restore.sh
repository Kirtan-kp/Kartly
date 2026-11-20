#!/usr/bin/env bash
# Restore the Kartly database from a backup made by scripts/backup.sh.
#   scripts/restore.sh FILE --yes
# This REPLACES the data in the database with the contents of FILE. Before it does, it stops the
# API and worker, and saves the current database as backups/pre-restore-*.dump so a mistaken
# restore can itself be undone. Afterwards it brings the schema to the latest revision, starts the
# API and worker again and checks they answer.
#
# Settings (environment): KARTLY_PG_EXEC, KARTLY_PG_USER, KARTLY_PG_DB as in backup.sh, and
#   KARTLY_STOP_CMD / KARTLY_START_CMD  commands that stop / start the API and worker
#   KARTLY_MIGRATE_CMD                  command that brings the schema to head
#   (defaults use docker-compose.deploy.yml)
set -euo pipefail
cd "$(dirname "$0")/.."

FILE="${1:-}"
CONFIRM="${2:-}"
PG_USER="${KARTLY_PG_USER:-kartly}"
PG_DB="${KARTLY_PG_DB:-kartly}"
DC="docker compose --env-file .env.deploy -f docker-compose.deploy.yml"
PG_EXEC="${KARTLY_PG_EXEC:-$DC exec -T postgres}"
STOP_CMD="${KARTLY_STOP_CMD:-$DC stop api worker}"
START_CMD="${KARTLY_START_CMD:-$DC up -d api worker}"
MIGRATE_CMD="${KARTLY_MIGRATE_CMD:-$DC run --rm migrate}"

if [ -z "$FILE" ] || [ "$CONFIRM" != "--yes" ]; then
  echo "usage: scripts/restore.sh FILE --yes" >&2
  echo "This replaces the database contents with FILE. Pass --yes to confirm." >&2
  exit 2
fi
[ -f "$FILE" ] || { echo "error: no such file: $FILE" >&2; exit 2; }
[ -s "$FILE" ] || { echo "error: $FILE is empty" >&2; exit 2; }

# shellcheck disable=SC2086  # PG_EXEC and the *_CMD values are command prefixes
$PG_EXEC pg_restore --list < "$FILE" > /dev/null || { echo "error: $FILE is not a readable backup" >&2; exit 2; }

SAFETY="(not taken yet)"
on_error() {
  echo >&2
  echo "RESTORE FAILED. The API and worker are still stopped, and the database may be half restored." >&2
  echo "Your data from before the restore is in: $SAFETY" >&2
  echo "Fix the cause and run the restore again, or restore that file the same way." >&2
}
trap on_error ERR

echo "==> stopping the API and worker"
$STOP_CMD

echo "==> saving the current database first"
SAFETY="$(KARTLY_BACKUP_PREFIX=pre-restore KARTLY_KEEP=1000000 KARTLY_PG_EXEC="$PG_EXEC" \
  KARTLY_PG_USER="$PG_USER" KARTLY_PG_DB="$PG_DB" scripts/backup.sh backups)"
echo "    saved as $SAFETY"

echo "==> restoring $FILE"
$PG_EXEC pg_restore -U "$PG_USER" -d "$PG_DB" --clean --if-exists --no-owner --exit-on-error < "$FILE"

echo "==> bringing the schema to the latest revision"
$MIGRATE_CMD

echo "==> starting the API and worker"
$START_CMD
echo "restored from $FILE"
