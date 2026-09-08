#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_root="${repo_root}/third_party/dexjoco/openpi"
target_root="${repo_root}/.local/external/simulation/s4_3_pi1/openpi"
patch_file="${repo_root}/patches/simulation/s4_3_pi1_openpi_hidden_loss_hook.patch"

if [[ -e "${target_root}" ]]; then
  if patch --dry-run -R -d "${target_root}" -p1 < "${patch_file}" >/dev/null; then
    printf 'validated existing patched writable OpenPI clone: %s\n' "${target_root}"
    exit 0
  fi
  printf 'refusing to replace unexpected existing target: %s\n' "${target_root}" >&2
  exit 1
fi

mkdir -p "$(dirname "${target_root}")"
cp -a "${source_root}" "${target_root}"
patch -d "${target_root}" -p1 < "${patch_file}"
printf 'prepared writable OpenPI clone: %s\n' "${target_root}"
