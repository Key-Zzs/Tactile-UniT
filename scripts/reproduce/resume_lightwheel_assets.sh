#!/usr/bin/env bash
# Resume the official RoboCasa Lightwheel asset download without deleting partial data.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_ROOT="$PROJECT_ROOT/third_party/robocasa-gr1-tabletop-tasks/robocasa/models/assets"
RUNTIME_ROOT="${LIGHTWHEEL_RUNTIME_DIR:-$PROJECT_ROOT/.local/tmp/lightwheel}"
PARTS_DIR="${LIGHTWHEEL_PARTS_DIR:-$RUNTIME_ROOT/parts}"
ARCHIVE="${LIGHTWHEEL_ARCHIVE:-$RUNTIME_ROOT/lightwheel.zip}"
REVISION="1b018839a6da865dffecd3185fe054211bc71270"
TOTAL=213480699
SHA256="8e42a0c835a205d7a0e7c33b5c33ab160b220fa9343b50c16526b7c712a2b6e2"
PART_COUNT=8
PROXY="${LIGHTWHEEL_PROXY:-}"
BASE_URL="https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets/resolve/${REVISION}/lightwheel.zip"

mkdir -p "$PARTS_DIR" "$ASSET_ROOT/objects"
if [[ -d "$ASSET_ROOT/objects/lightwheel" ]]; then
  echo "Already installed: $ASSET_ROOT/objects/lightwheel"
  exit 0
fi

curl_proxy=()
if [[ -n "$PROXY" ]]; then
  curl_proxy=(--proxy "$PROXY")
fi
signed_url=$(curl "${curl_proxy[@]}" -fsSIL --max-time 60 "$BASE_URL" | awk '
  BEGIN { IGNORECASE=1 }
  /^location:/ { sub(/^location:[[:space:]]*/, ""); sub(/\r$/, ""); url=$0 }
  END { print url }
')
[[ -n "$signed_url" ]] || { echo "Could not obtain a signed Hugging Face URL" >&2; exit 1; }

chunk=$(((TOTAL + PART_COUNT - 1) / PART_COUNT))
download_part() {
  local index="$1" start end expected part have next
  start=$((index * chunk))
  end=$((start + chunk - 1))
  (( end < TOTAL )) || end=$((TOTAL - 1))
  expected=$((end - start + 1))
  part="$PARTS_DIR/part.$index"
  have=0
  [[ -f "$part" ]] && have=$(stat -c %s "$part")
  if (( have > expected )); then
    echo "Invalid oversized partial part: $part ($have > $expected)" >&2
    return 1
  fi
  if (( have == expected )); then
    echo "part.$index complete ($have bytes)"
    return 0
  fi
  next=$((start + have))
  echo "part.$index resume: $have/$expected bytes"
  curl "${curl_proxy[@]}" -fL --retry 40 --retry-all-errors --retry-delay 2 \
    --connect-timeout 30 --max-time 0 -H "Range: bytes=$next-$end" "$signed_url" >> "$part"
  [[ $(stat -c %s "$part") == "$expected" ]] || {
    echo "Incomplete part.$index after curl" >&2
    return 1
  }
}

pids=()
for index in $(seq 0 $((PART_COUNT - 1))); do
  download_part "$index" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"

rm -f "$ARCHIVE".candidate
for index in $(seq 0 $((PART_COUNT - 1))); do
  cat "$PARTS_DIR/part.$index" >> "$ARCHIVE".candidate
done
[[ $(stat -c %s "$ARCHIVE".candidate) == "$TOTAL" ]]
echo "$SHA256  $ARCHIVE.candidate" | sha256sum -c -
unzip -t "$ARCHIVE".candidate >/dev/null
mv "$ARCHIVE".candidate "$ARCHIVE"
unzip -q "$ARCHIVE" -d "$ASSET_ROOT/objects"
[[ -d "$ASSET_ROOT/objects/lightwheel" ]]
echo "Installed official Lightwheel assets: $ASSET_ROOT/objects/lightwheel"
