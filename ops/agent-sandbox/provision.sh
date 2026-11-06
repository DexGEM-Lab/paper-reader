#!/usr/bin/env bash
# Operator-only: provision ONE shared sandbox, never called by a paper bot.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "usage: $0 IMAGE SESSIONS_ROOT PI_HOME LIMITED_LLM_ENV UID:GID" >&2
  exit 2
fi
image=$1
sessions=$(realpath -e "$2")
pi_home=$(realpath -e "$3")
llm_env=$(realpath -e "$4")
identity=$5
[[ $identity =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]] || { echo 'non-root numeric UID:GID required' >&2; exit 2; }
# Only one limited bot credential is accepted, never the Flask/Feishu/S3 env.
if grep -Ev '^($|#|PAPER_READER_LLM_API_KEY=.+$)' "$llm_env" >/dev/null; then
  echo 'LLM env may contain only PAPER_READER_LLM_API_KEY (plus blank lines/comments)' >&2
  exit 2
fi
if ! grep -q '^PAPER_READER_LLM_API_KEY=.' "$llm_env"; then
  echo 'PAPER_READER_LLM_API_KEY is required' >&2
  exit 2
fi
# Install/recheck the scoped host-network barrier before any bot can run.
"$(dirname "$0")/network.sh"
# No auto-replacement of a live container: stop/remove it explicitly for upgrades.
docker create --name paper-reader-agents --restart=no --init \
  --label ai.dexgem.paper-reader.sandbox=1 \
  --user "$identity" --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 2g --memory-swap 2g --cpus 2 --pids-limit 256 \
  --network paper-reader-agents --dns 1.1.1.1 --dns 8.8.8.8 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
  --mount "type=bind,source=$sessions,target=/workspaces" \
  --mount "type=bind,source=$pi_home,target=/pi-home,readonly" \
  --env-file "$llm_env" \
  "$image"
