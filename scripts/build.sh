#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${1:-deepseek-v4.1-flash-a100:dev}"
base_image="${BASE_IMAGE:-lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3}"
[[ "$base_image" == *@sha256:* ]] || {
  echo "BASE_IMAGE must be pinned by digest." >&2
  exit 1
}
mkdir -p runtime
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "Commit or isolate source changes before building." >&2
  exit 1
}
revision="$(git rev-parse HEAD)"
context="$(mktemp -d)"
trap 'rm -rf -- "$context"' EXIT
git archive HEAD | tar -xf - -C "$context"
docker build --network=none --pull=false \
  --build-arg "BASE_IMAGE=$base_image" --build-arg "SOURCE_REVISION=$revision" \
  -t "$image" "$context"
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
# Deliberately omit --gpus: these are source/CPU checks.
docker run --rm --runtime=runc --network none -e CUDA_VISIBLE_DEVICES= \
  -e NVIDIA_VISIBLE_DEVICES=void --entrypoint python3 "$image_id" \
  /opt/dsv41-a100/check_runtime.py
docker run --rm --runtime=runc --network none -e CUDA_VISIBLE_DEVICES= \
  -e NVIDIA_VISIBLE_DEVICES=void -e VLLM_TARGET_DEVICE=cpu \
  --entrypoint vllm "$image_id" serve --help \
  > runtime/vllm-serve-help.txt
docker image inspect --format '{{.Id}}' "$image_id" > runtime/built-image-id.txt
printf '%s\n' "$revision" > runtime/built-source-revision.txt
printf 'Built and CPU-checked %s\nBase image: %s\nGPU validation: NOT_RUN\n' \
  "$image_id" "$base_image"
