#!/usr/bin/env bash
# Download the source PDF from AWS (it is not committed: AWS owns the copyright). Any revision works for the
# CI tests; the retrieval regression guard is what fails if a revision changes retrieval quality.
set -euo pipefail

URL="https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/writing-best-practices-rag/writing-best-practices-rag.pdf"
PINNED="064e6abca87ddeec3d65f000d66ff42c55d6a4c232a7257fa4d6721479072bd6"  # revision the README numbers used
OUT="${1:-document.pdf}"

curl -fsSL --retry 3 -o "$OUT" "$URL"

if command -v sha256sum >/dev/null; then ACTUAL="$(sha256sum "$OUT" | cut -d' ' -f1)"
else ACTUAL="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"; fi

if [ "$ACTUAL" != "$PINNED" ]; then
  echo "note: AWS has revised the PDF ($ACTUAL); README numbers were measured on $PINNED" >&2
fi
echo "downloaded $OUT ($ACTUAL)"
