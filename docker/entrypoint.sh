#!/bin/sh
set -eu

if [ "${SKIP_STARTUP_CHECKS:-false}" != "true" ]; then
  echo "[entrypoint] Running startup network checks for Polymarket endpoints..."
  python - <<'PY'
import os
import socket
import sys

import requests

hosts = [
    "gamma-api.polymarket.com",
    "clob.polymarket.com",
    "data-api.polymarket.com",
]

for host in hosts:
    try:
        socket.gethostbyname(host)
        print(f"[startup-check] DNS OK: {host}")
    except Exception as exc:
        print(f"[startup-check] DNS FAIL: {host} -> {exc}", file=sys.stderr)
        sys.exit(1)

for url in [
    "https://gamma-api.polymarket.com/markets?limit=1",
    "https://docs.polymarket.com/api-reference/introduction",
]:
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code >= 400:
            print(f"[startup-check] HTTP FAIL: {url} -> {resp.status_code}", file=sys.stderr)
            sys.exit(1)
        print(f"[startup-check] HTTP OK: {url} -> {resp.status_code}")
    except Exception as exc:
        print(f"[startup-check] HTTP FAIL: {url} -> {exc}", file=sys.stderr)
        sys.exit(1)
PY
fi

echo "[entrypoint] Starting: $*"
exec "$@"
