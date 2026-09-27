#!/usr/bin/env bash
# From the laptop: copy the repo to the pod (tracked and new files, her memory and notes;
# no keys, no .archive, no models: the pod downloads its own).
#   bash deploy/runpod/push.sh <pod ip> <ssh port>
# The ip and port are on the pod's Connect tab ("SSH over exposed TCP").
set -euo pipefail
IP=$1
PORT=$2
KEY=${EVA_POD_KEY:-$HOME/.ssh/runpod_ed25519}
cd "$(dirname "$0")/../.."
TMP=$(mktemp -d)
{ git ls-files -co --exclude-standard | grep -v '^\.archive/'; ls memory.json notes.json 2>/dev/null || true; } >"$TMP/files"
tar -czf "$TMP/eva.tar.gz" -T "$TMP/files"
echo "$(wc -l <"$TMP/files") files, $(du -h "$TMP/eva.tar.gz" | cut -f1)"
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 root@$IP"
scp -q -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new "$TMP/eva.tar.gz" "root@$IP:/workspace/eva.tar.gz"
$SSH 'mkdir -p /workspace/eva && tar -xzf /workspace/eva.tar.gz -C /workspace/eva && echo "on the pod: /workspace/eva"'
rm -rf "$TMP"
