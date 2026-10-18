#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""reader of the transformer dump of a development run (NEIF_DEV_TFDUMP).
NOT part of the submission (dev/README.md).

One record per stream byte, taken before the byte is coded (src/neif_dev.h):
u32 position, u8 byte, u8 gate zone, u16 0, f16 slot[208] (the transformer
row the mixers read, set order), f16 hidden[192] (final hidden state),
f16 ppmd[208] (PPMd byte distribution, set order).

  python3 dev/tfdump.py <dump> [--cost <cost.u16>]

prints the cross entropy of the transformer slot and of PPMd on the dumped
bytes, outside and inside the gated zones, and, with the cost file of the same
run, that of the whole coder on the same bytes.

As a module: `records(path)` maps the file as a numpy record array.
"""

import argparse
import sys

import numpy as np

V = 208
D = 192
RECORD = np.dtype([
    ("pos", "<u4"), ("byte", "u1"), ("zone", "u1"), ("pad", "<u2"),
    ("slot", "<f2", (V,)), ("hidden", "<f2", (D,)), ("ppmd", "<f2", (V,)),
])
assert RECORD.itemsize == 1224

# the frozen byte set (src/vocab_frozen.h, transformer::kByteToIndex): a byte's
# index in a row is its rank in this sorted list
SET_BYTES = np.array(
    [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12] + list(range(32, 58)) + [64]
    + list(range(74, 84)) + [88] + list(range(91, 96)) + list(range(97, 123))
    + list(range(128, 256)), dtype=np.int64)
assert len(SET_BYTES) == V
INDEX_OF = np.full(256, -1, dtype=np.int64)
INDEX_OF[SET_BYTES] = np.arange(V)


def records(path):
    return np.memmap(path, dtype=RECORD, mode="r")


def nll_bits(rows, idx):
    """-log2 of the probability of the coded byte, rows renormalised"""
    p = rows.astype(np.float64)
    s = p.sum(axis=1)
    q = p[np.arange(len(idx)), idx] / np.maximum(s, 1e-300)
    return -np.log2(np.maximum(q, 1e-12))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--cost")
    ap.add_argument("--chunk", type=int, default=1 << 18)
    args = ap.parse_args(argv[1:])
    r = records(args.dump)
    n = len(r)
    if n == 0:
        sys.exit("empty dump")
    idx_all = INDEX_OF[r["byte"]]
    if (idx_all < 0).any():
        sys.exit("a byte outside the frozen set: 0x%02x"
                 % int(r["byte"][np.flatnonzero(idx_all < 0)[0]]))
    tf = np.zeros(n)
    pp = np.zeros(n)
    for lo in range(0, n, args.chunk):
        hi = min(n, lo + args.chunk)
        tf[lo:hi] = nll_bits(r["slot"][lo:hi], idx_all[lo:hi])
        pp[lo:hi] = nll_bits(r["ppmd"][lo:hi], idx_all[lo:hi])
    gated = r["zone"] != 0
    pos = r["pos"].astype(np.int64)
    print("%d records, stream positions %d .. %d" % (n, pos[0], pos[-1]))
    for name, m in (("all", np.ones(n, bool)), ("not gated", ~gated),
                    ("gated", gated)):
        if not m.any():
            continue
        line = "%-10s %10d bytes  transformer %.4f  ppmd %.4f bits/B" % (
            name, int(m.sum()), tf[m].mean(), pp[m].mean())
        print(line)
    if args.cost:
        c = np.fromfile(args.cost, dtype="<u2").reshape(-1, 8)
        if pos[-1] >= len(c):
            sys.exit("the cost file is shorter than the dump")
        coder = c[pos].sum(axis=1, dtype=np.float64) / 2048.0
        for name, m in (("all", np.ones(n, bool)), ("not gated", ~gated)):
            if m.any():
                print("%-10s coder %.4f bits/B (transformer alone %+.4f)"
                      % (name, coder[m].mean(), tf[m].mean() - coder[m].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
