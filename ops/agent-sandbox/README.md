# Shared paper-bot sandbox (FR-004)

## Architecture and boundary

All private/shared paper chats execute `docker exec` in **one** long-lived
`paper-reader-agents` container. Each conversation retains its own
`/workspaces/<session-slug>/cwd` and JSONL session; no container is created for a
paper or a turn. The Flask host process is the trusted controller. The container
has a non-root numeric UID/GID matching the service account, a read-only root,
no Linux capabilities, no-new-privileges, Docker's default seccomp profile, no
published ports, and shared CPU/memory/PID limits.

Only `<data>/agent-sessions` is writable from the container. A curated Pi profile
is mounted read-only at `/pi-home`. The library, SQLite databases, Flask/Feishu/S3
environment, host home, and Docker socket are **not mounted**. Materials are
copies, refreshed each turn; editing one cannot change the canonical paper.
Host-managed file writes use anchored directory descriptors and O_NOFOLLOW so
an agent-created symlink cannot redirect a later refresh into host files.

The dedicated IPv4-only bridge `paper-reader-agents` (`br-paperbots`) has scoped
INPUT and DOCKER-USER rules: drop host-local and private/link-local/special
network destinations; permit public egress, with public DNS resolvers. This
covers shell networking as well as search/fetch; a URL allowlist would not
constrain general bash. No global firewall flush or Docker restart is needed.

Pi loads the user's `pi-native-search` plugin at the exact configured revision
`62f26aec446bd87f018680c0fa459add3e0b660c`. Its active tools are explicitly
`read,bash,edit,write,grep,find,ls,web_search,web_fetch`. Personal/project
extensions, skills, prompt templates and image generation are not activated.
The system prompt and generated AGENTS.md explicitly allow general bash and
external research and require all agent-created changes to stay inside the
**initial cwd**, including after `cd` and through symlinks. External sources
must be identified by URL.

**This is host isolation, not adversarial separation between bots.** Bots share
an OS user, processes, the session export, resource budget, and limited LLM key.
The per-session modification boundary is the user-requested prompt rule; it is
not a kernel-enforced per-paper boundary. Another session's files are technically
accessible. Do not put host secrets or unrelated files in the export.
Docker access is root-equivalent for the trusted Flask controller; the socket
is never exposed to bots. A compromised web application is outside this
bot-execution boundary. The sandbox uses the host kernel, not a VM.

## Provisioning / deployment (operator, not the agent)

These steps change production and must be performed by the integrating operator.
Do not enable the new application revision before its shared runtime is ready.

1. Require Linux Docker Engine using its **iptables** firewall backend, public
   HTTPS/DNS egress, and the `docker` group. `network.sh` fails explicitly if
   DOCKER-USER is absent. On SELinux-enforcing hosts, label **only the two export
   directories** for container access (for example appropriate shared
   `container_file_t` fcontexts); do not disable SELinux or relabel all of `/opt`.
2. Build the image from the repository:
   ```sh
   docker build -t paper-reader-agents:fr004 ops/agent-sandbox
   ```
   Pi is pinned to the currently deployed version 0.82.1. The search plugin and
   its typebox dependency are pinned. If this workstation's Docker bridge cannot
   reach GitHub, `docker build --network host ...` is a **build-only** workaround;
   the runtime must keep its dedicated bridge.
3. Prepare paths (adjust data location if different):
   ```sh
   install -d -o paper-reader -g paper-reader -m 0750 /opt/paper-reader/data/agent-sessions
   install -d -o root -g root -m 0755 /opt/paper-reader/agent-pi-home
   install -m 0644 ops/agent-sandbox/pi-home/{settings,models}.json /opt/paper-reader/agent-pi-home/
   ```
   Do not copy `~/.pi/agent`, the old production Pi home, personal auth.json,
   shell-command API-key resolvers, or the application's full env into it.
   Set `PAPER_READER_PI_DIR=/opt/paper-reader/agent-pi-home` in the app env.
   The bundled profile uses `dexgem-messages` + `anthropic-messages` so the plugin
   can call native Claude search. Merely naming the old `litellm` provider with
   `openai-completions` routes to DDG, which failed from the test workstation.
4. Store **only** a dedicated budget/model-limited bot key in a root-0600 file,
   e.g. `/etc/paper-reader/agent-llm.env`:
   ```dotenv
   PAPER_READER_LLM_API_KEY=<limited-key>
   ```
   The key needs Claude Sonnet 5.5 Messages access **and** its server-side web
   search tool via `https://litellm.dex-gem.ai/v1/messages`. Bots can read their
   own key; its upstream budget/model restrictions are the spend boundary.
5. Run as root to create the single container and install scoped firewall rules:
   ```sh
   ops/agent-sandbox/provision.sh paper-reader-agents:fr004 \
     /opt/paper-reader/data/agent-sessions /opt/paper-reader/agent-pi-home \
     /etc/paper-reader/agent-llm.env "$(id -u paper-reader):$(id -g paper-reader)"
   ```
   `provision.sh` refuses a full application env. It creates, but does not start,
   the container. It never silently replaces an existing container.
6. Install `ops/systemd/paper-reader-agents.service` and updated
   `ops/systemd/paper-reader.service`. The latter grants Docker supplementary
   group access to the host controller and requires the shared runtime.
   ```sh
   systemctl daemon-reload
   systemctl enable --now paper-reader-agents.service
   # Restart paper-reader.service only as part of the coordinated app deploy.
   ```
   systemd owns runtime startup; Docker auto-restart is disabled so host firewall
   rules are reapplied **before** bots start after a reboot. Integrate
   `network.sh` into the host firewall's reload workflow too: a firewalld/iptables
   ruleset reset while the container runs must not remove this barrier. The app
   validates container/mount/resource configuration, not the host ruleset.
7. Runtime app settings:
   ```dotenv
   PAPER_READER_AGENT_CHAT_ENABLED=true
   PAPER_READER_PI_BIN=pi
   PAPER_READER_PI_DIR=/opt/paper-reader/agent-pi-home
   PAPER_READER_PI_SANDBOX_CONTAINER=paper-reader-agents
   PAPER_READER_PI_DOCKER_BIN=docker
   PAPER_READER_PI_TIMEOUT=300
   ```
   `PI_BIN` is an executable **inside the container**. A missing/stopped/unsafe
   sandbox fails the turn explicitly; there is no host execution fallback.

## Verification

Run ordinary unit/regression tests with `python -m pytest -q`.

Real sandbox tests are opt-in and create/remove temporary test containers on
the same dedicated network; they never mount production data or the repository:

```sh
sudo ops/agent-sandbox/network.sh
PAPER_READER_TEST_SANDBOX_IMAGE=paper-reader-agents:fr004 \
  python -m pytest -v tests/test_agent_sandbox_integration.py
# Optional: actual native-search and real-model tests; incurs LLM/search spend.
PAPER_READER_TEST_SANDBOX_IMAGE=paper-reader-agents:fr004 \
PAPER_READER_TEST_LLM_ENV=/path/to/limited-llm.env \
  python -m pytest -v tests/test_agent_sandbox_integration.py
```

Tests cover two cwds in the same container, persistent writes, canonical-source
integrity, read-only root/config, absent host sentinel/socket/secrets/process,
no mount capability, no-new-privileges, an all-interface host listener blocked
through the gateway, link-local egress blocked, public HTTPS, actual Pi tool
execution and session resume. The deterministic provider only scripts tool
selection, never search/fetch output. The credentialed tests additionally call
the actual plugin's native backend and ask a real paper bot to compute/write 42
with bash and find/cite the original Transformer paper.

A direct native-search probe is also available after provisioning:

```sh
# Use an existing session cwd, not /workspaces itself.
docker exec -w /workspaces/<session-slug>/cwd paper-reader-agents \
  node /opt/paper-reader/probe-search.mjs
```

It writes `search-probe.json` in that cwd and fails unless actual native search
returns an arxiv.org source. It does not change the Pi profile or call a generic
agent conversation. Results contain source text/URLs, not credentials.

## Deadlines, capacity and upgrades

The in-container Python supervisor is a Linux child subreaper. It owns the
per-turn deadline and kills/reaps descendants, including detached bash children,
on success, timeout or cancellation. The host watchdog cancels that supervisor;
killing a Docker exec client alone would leave work running. Tests exercise all
three child-cleanup paths. Do not start persistent daemons from paper tasks.
A host controller crash is still bounded by the supervisor's deadline.

The shared default budget is 2 CPUs, 2 GiB RAM, 256 PIDs and 256 MiB temporary
space. It is shared by all active bots; heavy tasks can contend or OOM another
turn. There is no per-paper quota. Writable notes/downloads count against host
data storage and backups; monitor growth. Material copies use more disk than the
old host symlinks, intentionally.

Upgrade: stop the reader and sandbox services, remove only `paper-reader-agents`,
rebuild/provision the replacement and start the sandbox before the reader. Never
restart shared Docker/LiteLLM services for this feature. Existing session slugs,
JSONL logs and cwd notes survive; each next turn refreshes materials as copies.
Do not restore old material symlinks into the export. Fixing poisoned arXiv HTML
caches is a separate data-quality task and is not accomplished by this sandbox.
