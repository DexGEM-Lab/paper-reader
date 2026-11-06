#!/usr/bin/env bash
# Operator/root only. Rules are scoped to this sandbox's dedicated bridge;
# never flush shared firewall tables/chains or restart Docker.
set -euo pipefail
network=paper-reader-agents
bridge=br-paperbots
chain=PAPER-READER-AGENTS
if ! docker network inspect "$network" >/dev/null 2>&1; then
  docker network create --driver bridge \
    --opt "com.docker.network.bridge.name=$bridge" "$network"
fi
[[ $(docker network inspect -f '{{.EnableIPv6}}' "$network") == false ]]
[[ $(docker network inspect -f '{{index .Options "com.docker.network.bridge.name"}}' "$network") == "$bridge" ]]
# Fail explicitly on an unsupported Docker firewall backend; never start with
# a silently ineffective rule. Docker's iptables backend supplies DOCKER-USER.
iptables -w -nL DOCKER-USER >/dev/null
iptables -w -N "$chain" 2>/dev/null || iptables -w -nL "$chain" >/dev/null
# Drop any host-local destination, including the host's public addresses.
iptables -w -C INPUT -i "$bridge" -j DROP 2>/dev/null || \
  iptables -w -I INPUT 1 -i "$bridge" -j DROP
# Drop private/link-local/special destinations, allow public internet egress.
for cidr in 0.0.0.0/8 10.0.0.0/8 100.64.0.0/10 127.0.0.0/8 169.254.0.0/16 \
            172.16.0.0/12 192.0.0.0/24 192.0.2.0/24 192.168.0.0/16 \
            198.18.0.0/15 198.51.100.0/24 203.0.113.0/24 224.0.0.0/4 240.0.0.0/4; do
  iptables -w -C "$chain" -d "$cidr" -j DROP 2>/dev/null || \
    iptables -w -A "$chain" -d "$cidr" -j DROP
done
iptables -w -C DOCKER-USER -i "$bridge" -j "$chain" 2>/dev/null || \
  iptables -w -I DOCKER-USER 1 -i "$bridge" -j "$chain"
echo 'paper-reader agent bridge: host/private destinations blocked; public egress enabled'
