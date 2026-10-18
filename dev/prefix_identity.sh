#!/bin/bash
# dev/prefix_identity.sh - do two runs code the same bits?  NOT part of the
# submission (dev/README.md).
#
#   bash dev/prefix_identity.sh <archive9 of run A> <archive9 of run B>
#
# Either file may belong to a run that is still going (comp9 -e writes the
# coded stream to archive9 in 1 MiB pieces, after the 5-byte header) or to a
# development run stopped with NEIF_DEV_STOP (its last bytes are the flush of
# the range coder).  The common prefix minus the last 8 bytes of the shorter
# file must be identical.
set -eu
A="${1:?usage: prefix_identity.sh <archive9 A> <archive9 B>}"
B="${2:?usage: prefix_identity.sh <archive9 A> <archive9 B>}"
la=$(stat -c %s "$A")
lb=$(stat -c %s "$B")
n=$(( (la < lb ? la : lb) - 8 ))
[ "$n" -gt 5 ] || { echo "too short to compare ($la, $lb bytes)" >&2; exit 2; }
if cmp -s -n "$n" "$A" "$B"; then
  echo "IDENTICAL on the first $n bytes ($((n - 5)) bytes of coded stream)"
else
  first=$(cmp -n "$n" "$A" "$B" | awk '{print $5}' | tr -d ',')
  echo "DIFFERENT: first difference at byte $first of $n"
  exit 1
fi
