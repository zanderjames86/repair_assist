#!/bin/bash
# Robust apt install for CI runners: GitHub's Ubuntu mirrors sometimes stall.
# Try the runner's existing package index first (fast), refresh it only if needed,
# and bound every attempt so a stalled mirror can't hang the job.
set -u
OPTS="-y -qq -o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 -o DPkg::Lock::Timeout=60"
export DEBIAN_FRONTEND=noninteractive
for attempt in 1 2 3; do
  if timeout 240 sudo -E apt-get install $OPTS "$@" < /dev/null; then exit 0; else echo "exit code $?"; fi
  echo "apt install attempt $attempt failed; refreshing package index"
  timeout 180 sudo -E apt-get update -qq -o Acquire::Retries=3 -o Acquire::http::Timeout=20 < /dev/null || true
done
exit 1
