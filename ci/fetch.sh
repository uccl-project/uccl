#!/usr/bin/env bash
set -euo pipefail
name=${1:?dependency name}
destination=${2:?destination}
read -r repository revision < <(python3 -c 'import json,sys; d=json.load(open("ci/config/dependencies.json"))[sys.argv[1]]; print(d["repository"],d["commit"])' "$name")
[[ $revision =~ ^[0-9a-f]{40}$ ]] || { echo 'Dependency must be pinned to a full commit' >&2; exit 1; }
mkdir -p "$destination"
curl --fail --location --retry 3 "https://codeload.github.com/$repository/tar.gz/$revision" | tar -xz --strip-components=1 -C "$destination"
