#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""a time profile from the samples of dev/prof_sample.sh.
NOT part of the submission (dev/README.md).

  python3 dev/prof_report.py <cmix_dev.map> <samples.txt> [--from P] [--to P]
                             [--top N]

<cmix_dev.map> is the linker map (lld -Map) of the build that ran
(build_dev.sh DEV_MAP=, kept by ab_prefix.sh); every function has its own
section (-ffunction-sections), so a pc maps to the function it lies in.  With
LTO most models are inlined into their callers (Predictor::Predict/Perceive,
FXCM::*); the transformer kernels are separate objects (qmat_dense.o, attn.o,
kda.o, ...), so the split by object file is the coarse profile.  --from/--to
keep the samples taken while the run's progress (percent of the coded
prefix) was in [P, P'): e.g. the article text of a 20 MB prefix from 44.22.
"""

import argparse
import bisect
import collections
import re
import subprocess
import sys


def load_map(path):
    iv = []
    for line in open(path):
        f = line.split()
        if len(f) >= 5 and "(.text" in f[4]:
            m = re.search(r"\(\.text(?:\.hot|\.unlikely|\.startup|\.split)?\.?(.*)\)$", f[4])
            name = m.group(1) if m and m.group(1) else f[4]
            iv.append((int(f[0], 16), int(f[2], 16), name, f[4].split(":")[0]))
    iv.sort()
    return iv


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("map")
    ap.add_argument("samples")
    ap.add_argument("--from", dest="lo", type=float, default=0.0)
    ap.add_argument("--to", dest="hi", type=float, default=101.0)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv[1:])
    iv = load_map(args.map)
    starts = [a for a, *_ in iv]
    fn = collections.Counter()
    obj = collections.Counter()
    n = outside = 0
    for line in open(args.samples):
        f = line.split()
        if len(f) < 3:
            continue
        try:
            prog, pc = float(f[1]), int(f[2], 16)
        except ValueError:
            continue
        if not args.lo <= prog < args.hi:
            continue
        n += 1
        i = bisect.bisect_right(starts, pc) - 1
        if i >= 0 and pc < iv[i][0] + iv[i][1]:
            fn[iv[i][2]] += 1
            obj[iv[i][3]] += 1
        else:
            outside += 1
    if not n:
        sys.exit("no samples in [%g, %g)" % (args.lo, args.hi))
    names = list(fn)
    try:
        out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True,
                             text=True).stdout.split("\n")
    except FileNotFoundError:
        out = names
    dem = dict(zip(names, out))
    print("%d samples with progress in [%g, %g), %d outside the map's code"
          % (n, args.lo, args.hi, outside))
    print("-- by object file")
    for o, c in obj.most_common():
        print("%5.1f%% %5d  %s" % (100.0 * c / n, c, o))
    print("-- by function")
    for name, c in fn.most_common(args.top):
        print("%5.1f%% %5d  %s" % (100.0 * c / n, c, dem.get(name, name)[:140]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
