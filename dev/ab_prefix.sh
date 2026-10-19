#!/bin/bash
# dev/ab_prefix.sh - A/B comparison of source trees on a prefix of the coder
# stream.  NOT part of the submission (dev/README.md).
#
#   bash dev/ab_prefix.sh <enwik9> <bytes> <workdir> <comp9> <arm> [<arm> ...]
#
# An arm is a source tree, or <tree>=<defines> for a tree built with further
# defines (`.=-DNEIF_ARM_X=1`, passed as DEV_DEFS to dev/build_dev.sh), with
# an optional @<container> at the end for another weight container
# (`.@new.c4`, passed as DEV_WEIGHTS).
# The first arm is the reference.  For each tree: dev/build_dev.sh builds its
# development binary (with the data blocks of <comp9>), then one run
#
#   NEIF_DEV_STOP=<bytes> NEIF_DEV_COST=cost.u16 cmix_dev -e enwik9 archive9
#
# codes the first <bytes> bytes of the stream in <workdir>/<n>/.  The runs go
# one after the other: each one takes the memory of a whole coder run.  The
# coded prefix is the same as the prefix of a full run, so the numbers are
# partial results of full runs, not of a model trained on a cut stream.
#
# Output: per tree the coded bytes of the prefix, the difference to the
# reference, the time of the coding loop and the peak RSS; then the costs
# per window against the reference (dev/cost_report.py).  Environment:
#   AB_WINDOW   window of the cost comparison in bytes (default 1000000)
#   AB_TIME=1   build the cycle-counter variant instead (time only; it may
#               code other bits than the submitted build)
#   AB_KEEP=1   keep the work files of the runs (stream copy, temp files)
#   AB_JOBS=k   run up to k arms at the same time (default 1); each run takes
#               one core and the memory of a coder run, see its peak RSS
#   AB_REF_DIR=<dir>  compare every arm with the cost file of an earlier
#               reference run in <dir> (a run directory of this script that
#               coded at least <bytes>; a prefix of a longer run codes the same
#               bits), instead of with the first arm
#   AB_REF_DUMPS=1  the first arm also writes the stream copy (stream.bin) and,
#               with AB_TFDUMP_FROM / AB_TFDUMP_LEN, a transformer dump
#               (tfdump.bin); both are kept
set -eu

usage="usage: ab_prefix.sh <enwik9> <bytes> <workdir> <comp9> <arm> [<arm> ...]"
ENWIK9="${1:?$usage}"
BYTES="${2:?$usage}"
WORKDIR="${3:?$usage}"
COMP9="${4:?$usage}"
shift 4
[ $# -ge 1 ] || { echo "$usage" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
ENWIK9="$(cd "$(dirname "$ENWIK9")" && pwd)/$(basename "$ENWIK9")"
[ -s "$ENWIK9" ] || { echo "ab_prefix.sh: no enwik9 at $ENWIK9" >&2; exit 2; }
case "$BYTES" in ''|*[!0-9]*) echo "ab_prefix.sh: <bytes> must be a number" >&2; exit 2;; esac
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
VARIANT=""
[ "${AB_TIME:-0}" = 1 ] && VARIANT=time

n=0
for arm in "$@"; do
  n=$((n + 1))
  weights=""
  case "$arm" in *@*) weights="${arm##*@}"; arm="${arm%@*}";; esac
  tree="${arm%%=*}"
  defs=""
  [ "$tree" != "$arm" ] && defs="${arm#*=}"
  d="$WORKDIR/$n"
  rm -rf "$d"
  mkdir -p "$d"
  echo "== build $n: $tree ${defs:+(defines: $defs)} ${weights:+(weights: $weights)}" >&2
  DEV_DEFS="$defs" DEV_WEIGHTS="$weights" DEV_MAP="$d/cmix_dev.map" \
    bash "$HERE/build_dev.sh" "$tree" "$COMP9" "$d/cmix_dev" $VARIANT >&2
  (cd "$tree" && git rev-parse HEAD 2>/dev/null || echo "not a git tree") > "$d/tree.txt"
  echo "$tree${defs:+ $defs}${weights:+ @$weights}" >> "$d/tree.txt"
  ln -s "$ENWIK9" "$d/enwik9"
done

# one run per arm, up to AB_JOBS at a time
run_arm() {
  local d="$WORKDIR/$1"
  local dumps=()
  if [ "$1" = 1 ] && [ "${AB_REF_DUMPS:-0}" = 1 ]; then
    dumps+=("NEIF_DEV_STREAM=$d/stream.bin")
    if [ -n "${AB_TFDUMP_LEN:-}" ]; then
      dumps+=("NEIF_DEV_TFDUMP=$d/tfdump.bin" "NEIF_DEV_TFDUMP_FROM=${AB_TFDUMP_FROM:-0}"
              "NEIF_DEV_TFDUMP_LEN=$AB_TFDUMP_LEN")
    fi
  fi
  (cd "$d" && env "${dumps[@]}" NEIF_DEV_STOP="$BYTES" NEIF_DEV_COST="$d/cost.u16" \
     ./cmix_dev -e enwik9 archive9 > run.out 2> run.err) || {
    echo "ab_prefix.sh: run $1 failed, see $d/run.err" >&2
    tail -20 "$d/run.err" >&2
    return 1
  }
  grep "^NEIF_DEV_RESULT" "$d/run.out" > "$d/result.txt" || {
    echo "ab_prefix.sh: run $1 printed no result, see $d/run.out" >&2; return 1; }
  if [ "${AB_KEEP:-0}" != 1 ]; then
    find "$d" -mindepth 1 -maxdepth 1 ! -name cost.u16 ! -name 'run.*' \
      ! -name result.txt ! -name tree.txt ! -name cmix_dev ! -name cmix_dev.map ! -name enwik9 \
      ! -name archive9 ! -name stream.bin ! -name tfdump.bin -exec rm -rf {} +
  fi
  echo "== run $1 done: $(cat "$d/result.txt")" >&2
}
JOBS="${AB_JOBS:-1}"
failed=0
running=0
for k in $(seq 1 "$n"); do
  if [ "$running" -ge "$JOBS" ]; then
    wait -n || failed=1
    running=$((running - 1))
  fi
  echo "== run $k started $(date -u +%T) UTC" >&2
  run_arm "$k" &
  running=$((running + 1))
done
while [ "$running" -gt 0 ]; do
  wait -n || failed=1
  running=$((running - 1))
done
[ "$failed" = 0 ] || { echo "ab_prefix.sh: at least one run failed" >&2; exit 1; }

python3 -I - "$WORKDIR" "$n" "${AB_REF_DIR:-}" "$BYTES" <<'EOF'
import os
import sys

import numpy as np

workdir, n, ref_dir, nbytes = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
rows = []
for k in range(1, n + 1):
    with open(os.path.join(workdir, str(k), "result.txt")) as f:
        fields = dict(x.split("=") for x in f.read().split()[1:])
    with open(os.path.join(workdir, str(k), "tree.txt")) as f:
        rev, tree = f.read().split("\n")[:2]
    rows.append((k, tree, rev[:12], fields))
if ref_dir:
    # the reference's coded size of this prefix: its own result if it coded
    # exactly this prefix, else the sum of its cost file plus the 5-byte header.
    # The u16 costs round small costs down: the sum is about 630 B short per
    # 20 MB, so the delta column then carries that bias; the cost comparison
    # below has none (both sides rounded alike).
    ref = None
    rpath = os.path.join(ref_dir, "result.txt")
    if os.path.exists(rpath):
        with open(rpath) as f:
            rf = dict(x.split("=") for x in f.read().split()[1:])
        if int(rf["stream_bytes"]) == nbytes:
            ref = int(rf["coded_bytes"])
            print("reference: %s, %d bytes for the first %d stream bytes (its result)"
                  % (ref_dir, ref, nbytes))
    if ref is None:
        c = np.fromfile(os.path.join(ref_dir, "cost.u16"), dtype="<u2").reshape(-1, 8)
        assert len(c) >= nbytes, "the reference run coded fewer bytes than this prefix"
        ref = int(round(c[:nbytes].sum(dtype=np.float64) / 2048.0 / 8.0)) + 5
        print("reference: %s, %d bytes for the first %d stream bytes (from its cost "
              "file, about 630 B per 20 MB short)" % (ref_dir, ref, nbytes))
else:
    ref = int(rows[0][3]["coded_bytes"])
print("%-3s %-12s %12s %12s %10s %9s %9s  %s" % (
    "n", "commit", "stream B", "coded B", "delta B", "delta %", "loop s",
    "tree"))
for k, tree, rev, f in rows:
    c = int(f["coded_bytes"])
    print("%-3d %-12s %12s %12d %+10d %+9.4f %9s  %s" % (
        k, rev, f["stream_bytes"], c, c - ref, 100.0 * (c - ref) / ref,
        f["seconds"], tree))
print("peak RSS (GiB): " + "  ".join(
    "%d: %.2f" % (k, int(f["peak_rss_kb"]) / 1048576.0) for k, _t, _r, f in rows))
EOF

REFCOST="${AB_REF_DIR:+$AB_REF_DIR/cost.u16}"
FIRST=2
[ -n "$REFCOST" ] && FIRST=1
REFCOST="${REFCOST:-$WORKDIR/1/cost.u16}"
if [ "$n" -ge "$FIRST" ]; then
  for k in $(seq "$FIRST" "$n"); do
    echo
    echo "== costs of $k against the reference, windows of ${AB_WINDOW:-1000000} B"
    python3 -I "$HERE/cost_report.py" --diff "$REFCOST" \
      "$WORKDIR/$k/cost.u16" --window "${AB_WINDOW:-1000000}" --top 8
  done
fi
