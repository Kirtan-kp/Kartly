#!/usr/bin/env bash
# Take a verified backup of the Kartly database.
#   scripts/backup.sh [DIRECTORY]        # default ./backups
# Writes kartly-YYYYmmdd-HHMMSS.dump (custom pg_dump format, mode 600), checks that it can be read
# back, and keeps the newest KARTLY_KEEP files (default 14). Copy the file off this machine: a
# backup that lives only next to the database does not survive losing the machine.
#
# Settings (environment):
#   KARTLY_PG_EXEC  prefix that runs a command inside the database container. Default: the
#                   compose file's postgres service. Tests point it at a local server.
#   KARTLY_PG_USER  (kartly)   KARTLY_PG_DB  (kartly)   KARTLY_KEEP  (14)
#   KARTLY_BACKUP_PREFIX (kartly) file name prefix; restore.sh uses pre-restore for its safety copy
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-backups}"
KEEP="${KARTLY_KEEP:-14}"
PG_USER="${KARTLY_PG_USER:-kartly}"
PG_DB="${KARTLY_PG_DB:-kartly}"
PREFIX="${KARTLY_BACKUP_PREFIX:-kartly}"
PG_EXEC="${KARTLY_PG_EXEC:-docker compose --env-file .env.deploy -f docker-compose.deploy.yml exec -T postgres}"

case "$KEEP" in '' | *[!0-9]*) echo "error: KARTLY_KEEP must be a number" >&2; exit 2 ;; esac
[ "$KEEP" -ge 1 ] || { echo "error: KARTLY_KEEP must be at least 1" >&2; exit 2; }

mkdir -p "$DEST"
chmod 700 "$DEST"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
TMP="$DEST/.$PREFIX-$STAMP.partial"
FINAL="$DEST/$PREFIX-$STAMP.dump"
# Never replace an existing backup, even one made in the same second.
n=1
while [ -e "$FINAL" ]; do
  FINAL="$DEST/$PREFIX-$STAMP-$n.dump"
  n=$((n + 1))
done
trap 'rm -f "$TMP"' EXIT

# shellcheck disable=SC2086  # PG_EXEC is a command prefix and must split into words
$PG_EXEC pg_dump -U "$PG_USER" -d "$PG_DB" --format=custom --no-owner > "$TMP"

[ -s "$TMP" ] || { echo "error: the dump is empty; no backup was made" >&2; exit 1; }
# A dump that cannot be listed cannot be restored: find that out now, not during an incident.
# shellcheck disable=SC2086
if ! $PG_EXEC pg_restore --list < "$TMP" > /dev/null; then
  echo "error: the dump could not be read back; no backup was made" >&2
  exit 1
fi

chmod 600 "$TMP"
mv "$TMP" "$FINAL"
trap - EXIT

# Keep the newest $KEEP backups with this prefix (names sort by time). Safety copies made before a
# restore have their own prefix, so they are never pruned here.
# shellcheck disable=SC2012
ls -1 "$DEST/$PREFIX"-*.dump 2>/dev/null | sort -r | tail -n +"$((KEEP + 1))" | while read -r old; do
  rm -f -- "$old"
done

echo "$FINAL"
