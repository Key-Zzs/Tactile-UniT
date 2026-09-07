#!/usr/bin/env bash
# Persistently resume the selectively mirrored official pi0.5 base checkpoint.

set -uo pipefail

repo_root="$(git rev-parse --show-toplevel)"
model_root="${repo_root}/.local/external/s4_3_pi0/models/DexJoCo-Pi05"
base_root="${model_root}/pi05_base"
expected_files=29
expected_bytes=12441749581
retry_seconds="${S4_3_DOWNLOAD_RETRY_SECONDS:-15}"
attempt=0

mkdir -p "${model_root}"

export HF_XET_CLIENT_RETRY_MAX_ATTEMPTS="${HF_XET_CLIENT_RETRY_MAX_ATTEMPTS:-50}"
export HF_XET_CLIENT_RETRY_MAX_DURATION="${HF_XET_CLIENT_RETRY_MAX_DURATION:-3600}"

# Direct access is substantially faster on the benchmark host. Set this switch
# to 1 only when a deployment requires its configured proxy.
if [[ "${S4_3_DOWNLOAD_USE_CONFIGURED_PROXY:-0}" != "1" ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi

tree_metrics() {
    local file_count total_bytes
    file_count="$(find "${base_root}" -type f 2>/dev/null | wc -l)"
    total_bytes="$(find "${base_root}" -type f -printf '%s\n' 2>/dev/null \
        | awk '{sum += $1} END {print sum + 0}')"
    printf '%s %s\n' "${file_count}" "${total_bytes}"
}

while true; do
    attempt=$((attempt + 1))
    printf '[%s] download attempt %d starting\n' "$(date --iso-8601=seconds)" "${attempt}"

    uvx --from huggingface_hub --with socksio \
        hf download DexJoCo/DexJoCo-Pi05 \
        --repo-type model \
        --include 'pi05_base/**' \
        --local-dir "${model_root}" \
        --max-workers 8
    command_status=$?

    read -r file_count total_bytes < <(tree_metrics)
    printf '[%s] attempt %d exit=%d final_files=%s/%s final_bytes=%s/%s\n' \
        "$(date --iso-8601=seconds)" "${attempt}" "${command_status}" \
        "${file_count}" "${expected_files}" "${total_bytes}" "${expected_bytes}"

    if [[ "${command_status}" -eq 0 \
        && "${file_count}" -eq "${expected_files}" \
        && "${total_bytes}" -eq "${expected_bytes}" ]]; then
        printf '[%s] official pi05_base download complete\n' "$(date --iso-8601=seconds)"
        exit 0
    fi

    printf '[%s] incomplete download; retrying in %s seconds\n' \
        "$(date --iso-8601=seconds)" "${retry_seconds}"
    sleep "${retry_seconds}"
done
