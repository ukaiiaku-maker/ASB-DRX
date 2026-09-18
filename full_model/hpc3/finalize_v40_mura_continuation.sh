#!/usr/bin/env bash
# Durable local checksum, extraction, and scientific classification follower.
set -Eeuo pipefail

: "${V40_FETCH_ROOT:?fetched continuation root required}"
: "${V40_HOMOGENEOUS_CASE:?current-source homogeneous case required}"
: "${V40_ANALYSIS_SOURCE:?analysis source worktree required}"
poll_s="${V40_POLL_SECONDS:-60}"
archive="$V40_FETCH_ROOT/results/mechanical_heterogeneity_to_5pct.tar.gz"
checksum="$V40_FETCH_ROOT/results/mechanical_heterogeneity_to_5pct.sha256"
decision_root="$V40_FETCH_ROOT/postprocessed"

while [[ ! -s "$archive" || ! -s "$checksum" ]]; do
  printf '%s waiting_for_verified_archive\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep "$poll_s"
done
expected=$(awk '{print $1}' "$checksum")
actual=$(shasum -a 256 "$archive" | awk '{print $1}')
printf 'archive=%s expected=%s actual=%s\n' "$archive" "$expected" "$actual"
test "$expected" = "$actual"

mkdir -p "$decision_root"
stage=$(mktemp -d "$decision_root/extracted.partial.XXXXXX")
tar -xzf "$archive" -C "$stage"
case_dir="$stage/mechanical_heterogeneity_topology_off_n64"
test -s "$case_dir/status.json"
PYTHONPATH="$V40_ANALYSIS_SOURCE/src:$V40_ANALYSIS_SOURCE" \
  python "$V40_ANALYSIS_SOURCE/full_model/analysis/postprocess_v40_one_grain.py" \
  --case-dir "$V40_HOMOGENEOUS_CASE" --case-dir "$case_dir" \
  --output "$decision_root/v40_one_grain_current_5pct_decision.json" \
  --figure "$decision_root/v40_one_grain_current_5pct_decision.png"
printf '%s\n' "$actual" >"$decision_root/archive.sha256.verified"
git -C "$V40_ANALYSIS_SOURCE" rev-parse HEAD \
  >"$decision_root/analysis_source_sha.txt"
mv "$stage" "$decision_root/extracted"
shasum -a 256 "$decision_root/v40_one_grain_current_5pct_decision.json" \
  >"$decision_root/v40_one_grain_current_5pct_decision.json.sha256"
