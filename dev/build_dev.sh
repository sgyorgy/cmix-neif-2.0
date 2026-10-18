#!/bin/bash
# dev/build_dev.sh - builds the development binary of a source tree: the
# compressor with the measurement hooks of src/neif_dev.h, followed by the
# data blocks of a built comp9, so that it runs `-e enwik9` like comp9.
# NOT part of the submission (dev/README.md).
#
#   bash dev/build_dev.sh <tree> <comp9> <out> [time]
#
#   <tree>   source tree in the layout of this package (this one, or a
#            `git worktree` of another commit)
#   <comp9>  a comp9 built by build.sh with the same dictionary hints,
#            article order and weight container as <tree>: its four data
#            blocks (hints, order, weights, header) are copied from its tail
#   <out>    the binary to write
#   time     build the variant with cycle counters (make cmix_devtime)
#
# DEV_DEFS in the environment (e.g. DEV_DEFS="-DNEIF_ARM_X=1") is passed to
# make: the defines of an experiment arm, added to every flag group.
#
# Like build.sh, the build runs in a copy of the tree under a clean
# environment (env -i); unlike it, nothing is packed, so the binary starts
# at once and its decompressor part is the unpacked image.
set -eu

TREE="$(cd "${1:?usage: build_dev.sh <tree> <comp9> <out> [time]}" && pwd)"
COMP9="${2:?usage: build_dev.sh <tree> <comp9> <out> [time]}"
OUT="${3:?usage: build_dev.sh <tree> <comp9> <out> [time]}"
TARGET=cmix_dev
[ "${4:-}" = time ] && TARGET=cmix_devtime
[ -s "$COMP9" ] || { echo "build_dev.sh: no comp9 at $COMP9" >&2; exit 2; }
[ -f "$TREE/makefile" ] || { echo "build_dev.sh: $TREE has no makefile" >&2; exit 2; }
grep -q "^$TARGET:" "$TREE/makefile" || {
  echo "build_dev.sh: the makefile of $TREE has no target $TARGET" >&2; exit 2; }
COMP9="$(cd "$(dirname "$COMP9")" && pwd)/$(basename "$COMP9")"
OUTDIR="$(cd "$(dirname "$OUT")" && pwd)"
OUT="$OUTDIR/$(basename "$OUT")"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/build_dev.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
(cd "$TREE" && tar cf - --exclude=./.build --exclude=./comp9 --exclude=./.git \
    --exclude=./dev/work .) | (cd "$WORK" && tar xf -)
chmod -R u+w "$WORK"
mkdir -p "$WORK/tmp"
(cd "$WORK" && env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME="$WORK" TMPDIR="$WORK/tmp" LC_ALL=C LANG=C TZ=UTC \
    make CC=clang++-18 CFLAGS_DEFINES="-DSEED=923 -DUPDATE_LIMIT=3000" \
    DEV_DEFS="${DEV_DEFS:-}" "$TARGET" -j > "$WORK/make.log" 2>&1) || {
  tail -30 "$WORK/make.log" >&2
  echo "build_dev.sh: make $TARGET failed" >&2
  exit 1
}

# The tail of comp9: hints, article order, weight container, then the 16-byte
# header with their sizes (HeaderInfo of src/readalike_prepr/self_extract.h).
python3 -I - "$COMP9" "$WORK/$TARGET" "$OUT" <<'EOF'
import struct
import sys

comp9, binary, out = sys.argv[1:4]
with open(comp9, "rb") as f:
    c9 = f.read()
d, o, z, w = struct.unpack("<4i", c9[-16:])
assert z == 0 and d > 0 and o > 0 and w > 0, (d, o, z, w)
assert d + o + w + 16 < len(c9), "comp9 is shorter than its blocks"
with open(binary, "rb") as f:
    image = f.read()
with open(out, "wb") as f:
    f.write(image + c9[-(d + o + w + 16):])
print("%s: %d B image + %d B blocks (hints %d, order %d, weights %d)"
      % (out, len(image), d + o + w + 16, d, o, w))
EOF
chmod 0755 "$OUT"
