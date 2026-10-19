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
# DEV_MAP=<path> keeps the linker map of the image (dev/prof_report.py).
# DEV_AUX=1 also builds the helper with the arm's defines (make cmix_devaux)
# and codes the hint package and the article order with it, in place of the
# blocks of <comp9>: needed by an arm that changes the plain CM (a mixer
# rule), which cannot decode the submitted blocks ("dictionary hints:
# derivation failed").  The helper allocates the full model: it must not
# run beside a coder run.
# DEV_WEIGHTS=<container> replaces the weight container: its FNV-1a-64 is
# stamped into the work copy of src/transformer.h (WeightsFingerprint) and it
# is appended in place of the container of <comp9>.
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
if [ -n "${DEV_WEIGHTS:-}" ]; then
  [ -s "$DEV_WEIGHTS" ] || { echo "build_dev.sh: no container at $DEV_WEIGHTS" >&2; exit 2; }
  DEV_WEIGHTS="$(cd "$(dirname "$DEV_WEIGHTS")" && pwd)/$(basename "$DEV_WEIGHTS")"
  python3 -I - "$DEV_WEIGHTS" "$WORK/src/transformer.h" <<'PY'
import re
import sys

container, header = sys.argv[1:3]
h = 0xcbf29ce484222325
with open(container, "rb") as f:
    for b in f.read():
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
src = open(header).read()
pat = r"(inline unsigned long long WeightsFingerprint\(\) \{\s*return )0x[0-9a-f]+ULL;"
new, n = re.subn(pat, r"\g<1>0x%016xULL;" % h, src)
assert n == 1, "WeightsFingerprint() not found in " + header
open(header, "w").write(new)
print("weights %s: fingerprint 0x%016x stamped" % (container, h))
PY
fi
(cd "$WORK" && env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME="$WORK" TMPDIR="$WORK/tmp" LC_ALL=C LANG=C TZ=UTC \
    make CC=clang++-18 CFLAGS_DEFINES="-DSEED=923 -DUPDATE_LIMIT=3000" \
    DEV_DEFS="${DEV_DEFS:-}" "$TARGET" -j > "$WORK/make.log" 2>&1) || {
  tail -30 "$WORK/make.log" >&2
  echo "build_dev.sh: make $TARGET failed" >&2
  exit 1
}
# DEV_MAP=<path>: keep the linker map of the image, for dev/prof_report.py
# (the data blocks appended below do not move the code)
[ -n "${DEV_MAP:-}" ] && cp "$WORK/cmix.map" "$DEV_MAP"

AUX_HINTS=""
AUX_ORDER=""
if [ "${DEV_AUX:-0}" = 1 ]; then
  (cd "$WORK" && env -i \
      PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      HOME="$WORK" TMPDIR="$WORK/tmp" LC_ALL=C LANG=C TZ=UTC \
      make CC=clang++-18 CFLAGS_DEFINES="-DSEED=923 -DUPDATE_LIMIT=3000" \
      DEV_DEFS="${DEV_DEFS:-}" cmix_devaux -j > "$WORK/make_aux.log" 2>&1) || {
    tail -30 "$WORK/make_aux.log" >&2
    echo "build_dev.sh: make cmix_devaux failed" >&2
    exit 1
  }
  (cd "$WORK" && ./cmix_devaux -c dictionary/english.hints comp_hints > aux.log 2>&1 &&
     ./cmix_devaux -c src/readalike_prepr/data/new_article_order comp_order >> aux.log 2>&1) || {
    tail -20 "$WORK/aux.log" >&2
    echo "build_dev.sh: the helper could not code the blocks" >&2
    exit 1
  }
  AUX_HINTS="$WORK/comp_hints"
  AUX_ORDER="$WORK/comp_order"
fi

# The tail of comp9: hints, article order, weight container, then the 16-byte
# header with their sizes (HeaderInfo of src/readalike_prepr/self_extract.h).
python3 -I - "$COMP9" "$WORK/$TARGET" "$OUT" "${DEV_WEIGHTS:-}" "$AUX_HINTS" "$AUX_ORDER" <<'EOF'
import struct
import sys

comp9, binary, out, weights, aux_hints, aux_order = sys.argv[1:7]
with open(comp9, "rb") as f:
    c9 = f.read()
d, o, z, w = struct.unpack("<4i", c9[-16:])
assert z == 0 and d > 0 and o > 0 and w > 0, (d, o, z, w)
assert d + o + w + 16 < len(c9), "comp9 is shorter than its blocks"
blocks = c9[-(d + o + w + 16):-(w + 16)]          # hints, order
if aux_hints:                                     # DEV_AUX=1: the arm's own blocks
    with open(aux_hints, "rb") as f:
        hints = f.read()
    with open(aux_order, "rb") as f:
        order = f.read()
    assert len(hints) >= 1024 and len(order) >= 1024, "the helper wrote empty blocks"
    print("blocks of the arm's helper: hints %d B (comp9 %d), order %d B (comp9 %d)"
          % (len(hints), d, len(order), o))
    blocks, d, o = hints + order, len(hints), len(order)
if weights:
    with open(weights, "rb") as f:
        container = f.read()
else:
    container = c9[-(w + 16):-16]
tail = blocks + container + struct.pack("<4i", d, o, 0, len(container))
with open(binary, "rb") as f:
    image = f.read()
with open(out, "wb") as f:
    f.write(image + tail)
print("%s: %d B image + %d B blocks (hints %d, order %d, weights %d)"
      % (out, len(image), len(tail), d, o, len(container)))
EOF
chmod 0755 "$OUT"
