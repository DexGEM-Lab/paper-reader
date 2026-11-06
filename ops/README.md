# ops — production deployment artifacts for reader.dex-gem.ai

Canonical copies of what runs on the DexGEM VPS (89.208.253.48, `ssh dexgem`).

- `systemd/paper-reader.service` → `/etc/systemd/system/` — gunicorn 1 worker / 8 threads on 127.0.0.1:8010, env from `/etc/paper-reader/paper-reader.env` + `/etc/paper-reader/litellm.env` (both root-protected, NOT in git).
- `systemd/paper-reader-backup.{service,timer}` + `backup-failure.service` → `/etc/systemd/system/` — daily 03:30 UTC snapshot to `s3://runtime/paper-reader/backups/snapshots/<ts>/` (endpoint https://oss.dex-gem.ai:1443, path-style). Credentials: `/etc/paper-reader/s3-backup.env` (root 0600, NOT in git).
- `backup-to-s3.sh` → `/opt/paper-reader/scripts/` — timestamped snapshots with SHA256SUMS/MANIFEST/COMPLETE; retention 14 daily + 8 weekly + 12 monthly.
- `nginx-reader.dex-gem.ai.conf` → `/etc/nginx/conf.d/` — internal TLS vhost (127.0.0.1:8443, wildcard origin cert). Also requires the SNI map entry `reader.dex-gem.ai dex_gem_web_https;` in `/etc/nginx/stream.d/dex-gem-sni.conf` and an explicit port-80 308 redirect.

## S3 credential renewal (weekly duty)

The oss-bot key lasts 168h. Renew before expiry: in Feishu group 🖥 算力-DexGEM, send
`@<oss-bot> oss-key /runtime/paper-reader/backups 168h paper-reader 每日快照备份` — the bot
auto-approves runtime ≤1w and DMs the new key to the requester. Install into
`/etc/paper-reader/s3-backup.env` on the VPS. Long-term fix: ask the storage platform
owners for a durable service account.

## Feedback agent

Set `PAPER_READER_FEISHU_FEEDBACK_CHAT_IDS` to comma-separated feedback group IDs and `PAPER_READER_INTERNAL_TOKEN` to a strong secret in the root-only app environment. Feedback chats also feed normal link import. The worker persists inbound messages, issue changes and outgoing threaded replies under the data library directory; transient model/API errors are retried. Gunicorn currently runs one worker, which is required for one feedback dispatcher.

Run `ops/feedback_watch.py --token-file <mode-0600-file> --cursor-file <private-state-file> --to <live-pi-session>` on the workstation; it polls every 180 seconds and advances its cursor only after `pi-notify` accepts the batch. The named Pi session must remain connected. Use an Authorization bearer header to call `/internal/feedback/pending?cursor=0`, or POST JSON `{ "issue": "ISSUE-001", "status": "fixed", "resolution_note": "...", "commit_ref": "..." }` to `/internal/feedback/resolve`. Updates and threaded replies are durably queued; `reply: queued` does not mean Feishu has delivered it yet. Check service logs for retries. Feature evaluation uses an `FR-NNN` identifier and the same endpoint.

The Feishu app needs tenant scopes `im:message:send_as_bot` for replies, `im:message:readonly` for fetching a parent outside the local ledger, and both `contact:contact.base:readonly` and `contact:user.base:readonly` (with appropriate contact data access) to resolve a sender's name. When a scope changes, restart the reader service to refresh its cached tenant token. If name lookup is unavailable, the ledger retains the sender's app-scoped open ID; if parent lookup fails, triage continues with available local context. Check the `jobs` and `outbox` tables for durable retry errors.
