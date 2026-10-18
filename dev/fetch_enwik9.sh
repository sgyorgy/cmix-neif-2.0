#!/bin/bash
# dev/fetch_enwik9.sh - fetches enwik9 and checks it by content.
# NOT part of the submission (dev/README.md).
#
#   bash dev/fetch_enwik9.sh <dir> [<url of enwik9.zip>]
#
# Without a URL it tries the official location
# (https://mattmahoney.net/dc/enwik9.zip).  The file is accepted only when its
# sha256 is that of enwik9 (1,000,000,000 bytes); the zip is unpacked with
# python's zipfile in its own directory and removed afterwards.
set -eu
DIR="${1:?usage: fetch_enwik9.sh <dir> [<url of enwik9.zip>]}"
URL="${2:-https://mattmahoney.net/dc/enwik9.zip}"
WANT=159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc
mkdir -p "$DIR"
DIR="$(cd "$DIR" && pwd)"
if [ -s "$DIR/enwik9" ] && [ "$(sha256sum "$DIR/enwik9" | cut -d' ' -f1)" = "$WANT" ]; then
  echo "enwik9 already in $DIR, sha256 ok"
  exit 0
fi
TMP="$(mktemp -d "$DIR/.fetch.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
echo "fetching $URL"
curl -fL --retry 4 --retry-delay 2 -o "$TMP/enwik9.zip" "$URL"
python3 -I - "$TMP/enwik9.zip" "$TMP" <<'EOF'
import sys
import zipfile

z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
assert names == ["enwik9"], "unexpected members: %r" % names[:5]
z.extract("enwik9", sys.argv[2])
EOF
GOT="$(sha256sum "$TMP/enwik9" | cut -d' ' -f1)"
[ "$GOT" = "$WANT" ] || { echo "sha256 mismatch: $GOT" >&2; exit 1; }
mv "$TMP/enwik9" "$DIR/enwik9"
echo "enwik9 in $DIR, $(wc -c < "$DIR/enwik9") bytes, sha256 ok"
