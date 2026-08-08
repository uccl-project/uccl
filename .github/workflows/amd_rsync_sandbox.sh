#!/usr/bin/env bash
# Run on amd0 (with forwarded SSH agent from CI) to rsync ~/uccl-ci-sandbox to the destination host.
set -euo pipefail

: "${DST_HOST:?DST_HOST required}"
: "${DST_USER:?DST_USER required}"

mkdir -p ~/.ssh
ssh-keyscan -H "$DST_HOST" >> ~/.ssh/known_hosts 2>/dev/null || true

rsync -az --exclude .git --delete \
  -e "ssh -o BatchMode=yes -o IdentitiesOnly=no -F /dev/null" \
  --rsync-path="mkdir -p ~/uccl-ci-sandbox && rsync" \
  ~/uccl-ci-sandbox/ \
  "${DST_USER}@${DST_HOST}:~/uccl-ci-sandbox/"
