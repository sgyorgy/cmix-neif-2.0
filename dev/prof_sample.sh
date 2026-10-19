#!/bin/bash
# dev/prof_sample.sh - a sampling time profile of a running coder, without a
# separate run.  NOT part of the submission (dev/README.md).
#
#   bash dev/prof_sample.sh <pid> <seconds> <run.err> <samples.txt>
#
# Every <seconds> seconds gdb attaches to <pid>, reads the program counter
# and detaches; each line of <samples.txt> is "unix time, the last progress
# value of <run.err>, pc".  The coder is stopped for a fraction of a second
# per sample, which costs wall time but does not change what it codes.  It
# ends when the process ends.  dev/prof_report.py maps the samples to
# functions with the linker map of the same build (ab_prefix.sh keeps it as
# cmix_dev.map next to each binary).
set -u
pid="${1:?usage: prof_sample.sh <pid> <seconds> <run.err> <samples.txt>}"
every="${2:?usage: prof_sample.sh <pid> <seconds> <run.err> <samples.txt>}"
err="${3:?usage: prof_sample.sh <pid> <seconds> <run.err> <samples.txt>}"
out="${4:?usage: prof_sample.sh <pid> <seconds> <run.err> <samples.txt>}"
command -v gdb > /dev/null || { echo "prof_sample.sh: gdb is missing" >&2; exit 2; }
while kill -0 "$pid" 2>/dev/null; do
  pc=$(gdb -p "$pid" -batch -nx -ex 'p/x $pc' 2>/dev/null | sed -n 's/^\$1 = //p')
  prog=$(grep '^progress' "$err" 2>/dev/null | tail -1 | awk '{print $2}')
  [ -n "$pc" ] && echo "$(date +%s) ${prog:-0} $pc" >> "$out"
  sleep "$every"
done
