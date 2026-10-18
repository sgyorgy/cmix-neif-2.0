#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""where the bits of the metadata block go (PLAN_10PCT.md 1.7).
NOT part of the submission (dev/README.md).

  python3 dev/meta_report.py <stream.bin> <cost.u16>

stream.bin is the stream copy of a development run (NEIF_DEV_STREAM), the cost
file the same run's NEIF_DEV_COST; the run must have coded past the block.
The block form is that of src/metablock_form.h: three head lines, then 243,425
lines of part A (page-id delta in decimal, the revid as four base-100 symbols,
then TAB and the restrictions where a page has them), then 243,425 lines of
part B in revid order (timestamp: decimal delta for records from 87,813 on,
fixed base-100 fields before; TAB name, TAB uid where present).
"""

import sys

import numpy as np

NN = 243425
HEAD_LINES = 3
OBEN = 87813


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    s = np.fromfile(argv[1], dtype=np.uint8)
    c = np.fromfile(argv[2], dtype="<u2").reshape(-1, 8)
    bits = c.sum(axis=1, dtype=np.float64) / 2048.0
    nl = np.flatnonzero(s == 0x0A)
    ab = int(nl[HEAD_LINES - 1]) + 1
    lines_end = nl[HEAD_LINES:HEAD_LINES + 2 * NN]       # newline of each line
    bis = int(lines_end[-1]) + 1
    if len(bits) < bis:
        sys.exit("the cost file covers %d bytes, the block ends at %d" % (len(bits), bis))
    starts = np.concatenate([[ab], lines_end[:-1] + 1])
    print("stream %d B; head lines %d B (%.0f B coded); block %d..%d, %d B, "
          "%.0f B coded" % (len(s), ab, bits[:ab].sum() / 8, ab, bis, bis - ab,
                            bits[ab:bis].sum() / 8))
    fields = {}

    def add(name, lo, hi):
        if hi > lo:
            f = fields.setdefault(name, [0, 0.0, 0])
            f[0] += hi - lo
            f[1] += bits[lo:hi].sum()
            f[2] += 1

    tab = 0x09
    redirect = 0
    for i in range(NN):                                  # part A
        lo, end = int(starts[i]), int(lines_end[i])
        line = s[lo:end]
        t = np.flatnonzero(line == tab)
        rev_end = lo + (int(t[0]) if len(t) else len(line))
        add("A page-id delta", lo, rev_end - 4)
        add("A revid (4 base-100)", rev_end - 4, rev_end)
        add("A restrictions", rev_end, end)
        add("A newline", end, end + 1)
    for k in range(NN):                                  # part B
        lo, end = int(starts[NN + k]), int(lines_end[NN + k])
        line = s[lo:end]
        t = np.flatnonzero(line == tab)
        f0 = lo + (int(t[0]) if len(t) else len(line))
        add("B timestamp delta" if k >= OBEN else "B timestamp fixed", lo, f0)
        if len(t) >= 2:
            add("B name", f0, lo + int(t[1]))
            add("B uid", lo + int(t[1]), end)
        else:
            add("B name", f0, end)
        add("B newline", end, end + 1)
    total = sum(f[1] for f in fields.values())
    print("%-22s %10s %10s %12s %10s %9s" % ("field", "records", "bytes", "coded B",
                                             "bits/rec", "share"))
    for name, (nbytes, b, nrec) in fields.items():
        print("%-22s %10d %10d %12.0f %10.2f %8.1f%%" % (
            name, nrec, nbytes, b / 8, b / max(nrec, 1), 100 * b / total))
    print("%-22s %10s %10s %12.0f" % ("total", "", "", total / 8))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
