#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SOURCE_CHECKPOINT DESTINATION_CHECKPOINT" >&2
  exit 2
fi

source_checkpoint=$1
destination_checkpoint=$2
destination_parent=$(dirname "${destination_checkpoint}")

test -d "${source_checkpoint}/params"
test -d "${source_checkpoint}/assets"
test ! -e "${destination_checkpoint}"

mkdir -p "${destination_parent}"
test "$(stat -c %d "${source_checkpoint}")" = "$(stat -c %d "${destination_parent}")"

cp -al "${source_checkpoint}" "${destination_checkpoint}"
diff -u \
  <(find "${source_checkpoint}" -type f -printf '%P %i %s\n' | sort) \
  <(find "${destination_checkpoint}" -type f -printf '%P %i %s\n' | sort)
