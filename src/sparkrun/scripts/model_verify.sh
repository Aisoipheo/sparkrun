#!/bin/bash
set -uo pipefail
# Verify a cached HF model against the cache's own content addressing: weight
# blobs are LFS files whose blob filename IS the sha256 of their contents, so
# hashing them needs no network access and no manifest.  Presence-only checks
# (any weight file found) cannot see a truncated or raced transfer -- the
# snapshot symlink and the blob both exist while the bytes are wrong.
#
# Exit codes: 0 = verified; 1 = missing/corrupt (with SPARKRUN_VERIFY_REPAIR=1,
# corrupt blobs are removed first so a re-download refetches them).
#
# Placeholders filled by Python: {cache_path}, {revision}
# NOTE: this file is consumed via Python str.format(); it must contain NO
# literal curly-brace characters.

CACHE_PATH="{cache_path}"
MODEL_REVISION={revision}
MARKER="$CACHE_PATH/.sparkrun-verified"

# sparkrun:include _hf_snapshots.sh

# Steady-state fast path: a verification marker younger than every weight
# blob means nothing changed since the last clean hash pass -- skip the
# expensive hashing.  Any blob written afterwards (fresh mtime) invalidates
# the marker, so a raced or repaired cache is always re-hashed.
if [ -f "$MARKER" ]; then
    STALE="$(find -L "$CACHE_PATH" \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.gguf" \) -type f -newer "$MARKER" -print -quit 2>/dev/null)"
    if [ -z "$STALE" ]; then
        echo "verified marker is fresh; skipping hash pass"
        exit 0
    fi
    echo "cache changed since last verification; re-hashing"
fi

FOUND=0
FAIL=0

while IFS= read -r SNAPSHOT_DIR; do
    [ -n "$SNAPSHOT_DIR" ] || continue
    # -L so -type f follows the link, mirroring model_sync.sh: snapshot
    # entries are symlinks.  Note that find -L silently skips dangling
    # entries -- a snapshot whose links are ALL dangling fails below via
    # FOUND=0, but a partially dangling one is invisible to this scan.
    while IFS= read -r -d '' f; do
        FOUND=$((FOUND + 1))
        blob="$(readlink -f "$f" 2>/dev/null)"
        if [ ! -f "$blob" ]; then
            echo "missing blob for snapshot entry: $f"
            FAIL=$((FAIL + 1))
            continue
        fi
        sum="$(sha256sum "$blob" 2>/dev/null | cut -d" " -f1)"
        if [ "$sum" != "$(basename "$blob")" ]; then
            echo "checksum mismatch: $blob"
            if [ "$(printenv SPARKRUN_VERIFY_REPAIR)" = "1" ]; then
                rm -f "$blob"
                echo "removed corrupt blob: $blob"
            fi
            FAIL=$((FAIL + 1))
        fi
    done < <(find -L "$SNAPSHOT_DIR" \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.gguf" \) -type f -print0 2>/dev/null)
done <<<"$(sparkrun_hf_snapshot_dirs "$CACHE_PATH" "$MODEL_REVISION")"

if [ "$FOUND" -eq 0 ]; then
    echo "no weight files found under $CACHE_PATH"
    exit 1
fi
if [ "$FAIL" -gt 0 ]; then
    echo "verification failed: $FAIL problem(s) across $FOUND weight file(s)"
    exit 1
fi
touch "$MARKER"
echo "verified: $FOUND weight file(s) match their checksums"
exit 0
