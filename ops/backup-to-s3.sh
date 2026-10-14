#!/usr/bin/env bash
# Create a complete, checksummed paper-reader data snapshot and publish it to S3.
set -euo pipefail

STATE=/opt/paper-reader/data
DB_REL='library/.paper-reader-team.db'
DB="$STATE/$DB_REL"
ENV_FILE=/etc/paper-reader/s3-backup.env
AWS_CONFIG_FILE=/etc/paper-reader/aws-config
DEST_BUCKET=runtime
DEST_PREFIX='paper-reader/backups'
SNAPSHOTS_PREFIX="$DEST_PREFIX/snapshots"
STAGING_ROOT=/opt/paper-reader/backups
DRY_RUN=0

usage() {
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
}

if [[ "${1:-}" == '--dry-run' ]]; then
  DRY_RUN=1
  shift
fi
if (( $# != 0 )); then
  usage
  exit 64
fi

for required in aws sqlite3 rsync tar sha256sum date flock; do
  command -v "$required" >/dev/null || {
    printf 'Required command is unavailable: %s\n' "$required" >&2
    exit 69
  }
done
[[ -d "$STATE" ]] || { printf 'Data tree is absent: %s\n' "$STATE" >&2; exit 66; }
[[ -f "$DB" ]] || { printf 'SQLite database is absent: %s\n' "$DB" >&2; exit 66; }

# Prevent overlapping timers/manual runs. This does not lock application writes.
exec 9>"$STAGING_ROOT/.backup-to-s3.lock"
flock -n 9 || { printf 'Another paper-reader backup is already running.\n' >&2; exit 75; }

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
stage="$(mktemp -d "$STAGING_ROOT/.s3-backup.XXXXXX")"
cleanup() { rm -rf "$stage"; }
trap cleanup EXIT
mkdir -p "$(dirname "$stage/data/$DB_REL")"

# ProtectSystem=strict makes the live DB directory read-only for this unit,
# so SQLite cannot create WAL sidecars to open the live DB. Copy DB+WAL+SHM
# into writable staging (SQLite recovers WAL on open), then .backup from there.
cp -a "$DB" "$stage/team.db.live"
for sidecar in "$DB-wal" "$DB-shm"; do
  [[ -f "$sidecar" ]] && cp -a "$sidecar" "$stage/team.db.live${sidecar#$DB}"
done
sqlite3 "$stage/team.db.live" ".timeout 5000" ".backup '$stage/team.db.snapshot'"
rm -f "$stage/team.db.live" "$stage/team.db.live-wal" "$stage/team.db.live-shm"
rsync -a --delete \
  --exclude="$DB_REL" \
  --exclude="$DB_REL-wal" \
  --exclude="$DB_REL-shm" \
  --exclude="$DB_REL-journal" \
  --exclude='*.env' \
  --exclude='.env' \
  --exclude='secrets/***' \
  "$STATE/" "$stage/data/"

# Recreate the checked DB path in the copied data tree after rsync (--delete
# would otherwise drop it; the snapshot lives outside data/ until this point).
cp -a "$stage/team.db.snapshot" "$stage/data/$DB_REL"
tar --zstd -cf "$stage/data.tar.zst" -C "$stage" data
(
  cd "$stage"
  sha256sum data.tar.zst > SHA256SUMS
  cat >MANIFEST <<EOF
schema=paper-reader-backup-v1
created_at=$stamp
data_root=/opt/paper-reader/data
database_path=$DB_REL
archive=data.tar.zst
excluded=*.env,.env,secrets/***
EOF
  sha256sum -c SHA256SUMS
)

if (( DRY_RUN )); then
  sqlite3 "$stage/data/$DB_REL" 'PRAGMA integrity_check;' | grep -qx 'ok'
  if tar --list --file "$stage/data.tar.zst" | grep -Eq '(^|/)(secrets/|[^/]*\.env$)'; then
    printf 'Excluded secret material unexpectedly entered the archive.\n' >&2
    exit 65
  fi
  printf 'Dry-run snapshot assembled at %s (will be removed on exit).\n' "$stage"
  printf '%s\n' 'Archive contents:'
  tar --list --file "$stage/data.tar.zst" | sed -n '1,80p'
  printf '%s\n' 'Manifest:'
  cat "$stage/MANIFEST"
  exit 0
fi

[[ -r "$ENV_FILE" ]] || { printf 'Missing credential file: %s\n' "$ENV_FILE" >&2; exit 78; }
# shellcheck source=/etc/paper-reader/s3-backup.env
source "$ENV_FILE"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
: "${AWS_ENDPOINT_URL:=https://oss.dex-gem.ai:1443}"
if [[ "$AWS_ACCESS_KEY_ID" == __REQUEST_PENDING__ || "$AWS_SECRET_ACCESS_KEY" == __REQUEST_PENDING__ ]]; then
  printf 'S3 credentials are still pending in %s.\n' "$ENV_FILE" >&2
  exit 78
fi
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION AWS_ENDPOINT_URL AWS_CONFIG_FILE

snapshot_uri="s3://$DEST_BUCKET/$SNAPSHOTS_PREFIX/$stamp/"
aws s3 cp "$stage/" "$snapshot_uri" --recursive --endpoint-url "$AWS_ENDPOINT_URL" --only-show-errors
# COMPLETE is intentionally the last object: restore selection only considers it.
printf '%s\n' "$stamp" | aws s3 cp - "${snapshot_uri}COMPLETE" --endpoint-url "$AWS_ENDPOINT_URL" --only-show-errors

# Retain the 14 newest daily snapshots, plus 8 Sunday snapshots and 12 snapshots
# made on the first day of their month. Only COMPLETE snapshots are eligible.
mapfile -t complete_keys < <(
  aws s3api list-objects-v2 \
    --bucket "$DEST_BUCKET" \
    --prefix "$SNAPSHOTS_PREFIX/" \
    --endpoint-url "$AWS_ENDPOINT_URL" \
    --query 'Contents[?ends_with(Key, `/COMPLETE`)].Key' \
    --output text | tr '\t' '\n' | sort
)
declare -A keep=() seen_day=() seen_sunday=() seen_month=()
declare -a ordered=()
for key in "${complete_keys[@]}"; do
  [[ "$key" =~ ^${SNAPSHOTS_PREFIX}/([0-9]{8}T[0-9]{6}Z)/COMPLETE$ ]] || continue
  snapshot="${BASH_REMATCH[1]}"
  day="${snapshot:0:8}"
  # Iterate newest-first by prepending; the keys themselves are chronological.
  ordered+=("$snapshot")
done
for ((i=${#ordered[@]}-1; i>=0; i--)); do
  snapshot="${ordered[i]}"
  day="${snapshot:0:8}"
  if (( ${#seen_day[@]} < 14 )) && [[ -z "${seen_day[$day]+x}" ]]; then
    keep[$snapshot]=1
    seen_day[$day]=1
  fi
  dow="$(date -u -d "${day:0:4}-${day:4:2}-${day:6:2}" +%u)"
  if [[ "$dow" == 7 ]] && (( ${#seen_sunday[@]} < 8 )) && [[ -z "${seen_sunday[$day]+x}" ]]; then
    keep[$snapshot]=1
    seen_sunday[$day]=1
  fi
  month="${day:0:6}"
  if [[ "${day:6:2}" == 01 ]] && (( ${#seen_month[@]} < 12 )) && [[ -z "${seen_month[$month]+x}" ]]; then
    keep[$snapshot]=1
    seen_month[$month]=1
  fi
done
for snapshot in "${ordered[@]}"; do
  if [[ -z "${keep[$snapshot]+x}" ]]; then
    aws s3 rm "s3://$DEST_BUCKET/$SNAPSHOTS_PREFIX/$snapshot/" --recursive \
      --endpoint-url "$AWS_ENDPOINT_URL" --only-show-errors
    printf 'Pruned snapshot %s\n' "$snapshot"
  fi
done
printf 'Completed snapshot %s\n' "$snapshot_uri"
