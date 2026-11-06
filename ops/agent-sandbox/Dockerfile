FROM node:22.22.0-bookworm-slim

ARG PI_VERSION=0.82.1
ARG SEARCH_REF=62f26aec446bd87f018680c0fa459add3e0b660c
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl git jq poppler-utils python3 python3-pip python3-venv ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global @earendil-works/pi-coding-agent@${PI_VERSION} \
    && npm install --prefix /opt/search-deps @sinclair/typebox@0.34.41 \
    && rm -rf /root/.npm
RUN git clone https://github.com/yiwenlu66/pi-native-search.git /opt/pi-native-search \
    && cd /opt/pi-native-search && git checkout "$SEARCH_REF" \
    && ln -s /opt/search-deps/node_modules /opt/pi-native-search/node_modules \
    && rm -rf /opt/pi-native-search/.git

COPY run-turn.py probe-search.mjs bash-alias.mjs /opt/paper-reader/
ENV PI_CODING_AGENT_DIR=/pi-home PI_SKIP_VERSION_CHECK=1
# Runtime overrides with the numeric paper-reader service uid:gid.
USER 1000:1000
WORKDIR /workspaces
CMD ["sleep", "infinity"]
