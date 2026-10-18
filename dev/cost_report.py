#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""costs of a development run (NEIF_DEV_COST) per window, or the difference
of two runs.  NOT part of the submission (dev/README.md).

The cost file holds per stream byte 8 x u16, the coded cost of each bit
(high bit first) in 1/2048 bit, from the state of the range coder.

  python3 dev/cost_report.py <cost.u16> [--window B] [--stream <copy>]
  python3 dev/cost_report.py --diff <ref.u16> <arm.u16> [--window B] [--top N]

--stream adds a split by byte class (the stream copy of NEIF_DEV_STREAM):
the bytes of the frozen alphabet are text and markup of the preprocessed
stream, so the classes are the code bytes of the stream form, not letters.
"""

import argparse
import sys

import numpy as np

SCALE = 2048.0


def load(path):
    a = np.fromfile(path, dtype="<u2")
    if a.size % 8:
        sys.exit("%s: %d values, not a multiple of 8" % (path, a.size))
    return a.reshape(-1, 8)


def byte_bits(costs):
    return costs.sum(axis=1, dtype=np.float64) / SCALE


def windows(bits, window):
    n = (len(bits) + window - 1) // window
    edges = np.minimum(np.arange(n + 1) * window, len(bits))
    sums = np.add.reduceat(bits, edges[:-1]) if len(bits) else np.zeros(0)
    return edges, sums


def report(path, window, stream_path):
    costs = load(path)
    bits = byte_bits(costs)
    total = bits.sum()
    print("%s: %d stream bytes, %.0f B coded (%.6f bits per byte)"
          % (path, len(bits), total / 8, total / max(len(bits), 1)))
    per_bit = costs.sum(axis=0, dtype=np.float64) / SCALE / 8
    print("bytes per bit position (high bit first): "
          + " ".join("%.0f" % x for x in per_bit))
    edges, sums = windows(bits, window)
    print("%12s %12s %12s %10s" % ("from", "to", "coded B", "bits/B"))
    for k in range(len(sums)):
        lo, hi = edges[k], edges[k + 1]
        print("%12d %12d %12.0f %10.4f" % (lo, hi, sums[k] / 8,
                                           sums[k] / max(hi - lo, 1)))
    if stream_path:
        s = np.fromfile(stream_path, dtype=np.uint8)[:len(bits)]
        print("per byte value (top 24 by coded bytes):")
        tot = np.bincount(s, weights=bits[:len(s)], minlength=256)
        cnt = np.bincount(s, minlength=256)
        for v in np.argsort(-tot)[:24]:
            print("  0x%02x %10d bytes %12.0f B coded %8.4f bits/B"
                  % (v, cnt[v], tot[v] / 8, tot[v] / max(cnt[v], 1)))


def diff(ref_path, arm_path, window, top):
    a = byte_bits(load(ref_path))
    b = byte_bits(load(arm_path))
    n = min(len(a), len(b))
    if len(a) != len(b):
        print("note: %d and %d stream bytes, compared on the first %d"
              % (len(a), len(b), n))
    a, b = a[:n], b[:n]
    d = b - a
    print("reference %.0f B, arm %.0f B, difference %+.0f B (%+.4f %%)"
          % (a.sum() / 8, b.sum() / 8, d.sum() / 8,
             100.0 * d.sum() / max(a.sum(), 1e-9)))
    edges, sums = windows(d, window)
    _e, ref_sums = windows(a, window)
    print("%12s %12s %12s %10s" % ("from", "to", "delta B", "delta %"))
    for k in range(len(sums)):
        print("%12d %12d %+12.0f %+10.4f" % (
            edges[k], edges[k + 1], sums[k] / 8,
            100.0 * sums[k] / max(ref_sums[k], 1e-9)))
    if top and len(sums) > 1:
        order = np.argsort(sums)
        print("largest gains:  " + "  ".join(
            "%d..%d %+.0f" % (edges[k], edges[k + 1], sums[k] / 8)
            for k in order[:top] if sums[k] < 0))
        print("largest losses: " + "  ".join(
            "%d..%d %+.0f" % (edges[k], edges[k + 1], sums[k] / 8)
            for k in order[::-1][:top] if sums[k] > 0))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--window", type=int, default=1000000)
    ap.add_argument("--stream")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args(argv[1:])
    if args.diff:
        if len(args.files) != 2:
            ap.error("--diff takes two cost files")
        diff(args.files[0], args.files[1], args.window, args.top)
    else:
        if len(args.files) != 1:
            ap.error("one cost file, or --diff with two")
        report(args.files[0], args.window, args.stream)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
