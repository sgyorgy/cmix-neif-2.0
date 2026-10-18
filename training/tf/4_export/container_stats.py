#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Derived work.  The container formats this script reads (magic FX2TFWC2 to
# FX2TFWC4), the range coder and the bit trees are `weights_compress_v3.py`
# next to this file, which is itself a fork of `pysrc/weights_compress.py` from
# fx2-cmix-transformer by Vladimir Ivanov (GPL-3.0).  Our addition is this
# script.  See ../../ATTRIBUTION.md.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.  See the LICENSE file of this entry for
# the full text.

"""level statistics of the int4 tensors of a weight container

Reads a container in memory (nothing is written) and prints, per tensor
family ('blocks.<n>.' folded to 'blocks.#.'), the number of quantized weights,
their order-0 entropy, the share of the two clip levels -7 and +7 and the
share of level 0; then the totals:

  * the sum of the per-tensor order-0 entropies;
  * the entropy of the global histogram, i.e. what ONE static tree for all
    tensors would cost; v3/v4 code the int4 levels with one counting tree
    (halved at 1023), which lands between the two, so the difference bounds
    what per-family trees could save;
  * what the rest of the container (scales, f32 tensors, names) costs.

Purpose: the rate side of a rate-distortion change of the quantizer.  On the
shipped container (7,785,008 int4 weights) the levels cost 3.73 bits per
weight and 6.7 % of the weights sit on the clip levels; that is the figure a
re-quantization has to move, while per-family trees save only a few KB.

Deterministic, no randomness, no network, no environment, torch free (the
stand in for `pysrc.export_weights` of prune_dead_neurons.py is used when
torch is not installed).

Usage, from the tree root:

  python3 tf/4_export/container_stats.py <container.c4>
"""

import os
import re
import sys

sys.dont_write_bytecode = True   # leave no __pycache__ in the package

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# installs the torch free stand in for pysrc.export_weights when needed
import prune_dead_neurons  # noqa: E402,F401
import weights_compress_v3 as wc3  # noqa: E402

LEVELS = 15                      # int4 weights in [-7, 7]
DTYPE_INT8 = 0


def entropy_bits(hist):
    """order-0 entropy of a histogram, bits per symbol"""
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def family(name):
    """'blocks.<n>.' folded to 'blocks.#.', the '.q' suffix dropped"""
    g = re.sub(r"blocks\.\d+\.", "blocks.#.", name)
    return re.sub(r"\.(weight\.q|q)$", "", g)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = argv[1]
    tensors = wc3.read_tensor_file_v3(path)

    families = {}                # family -> [weights, entropy bits, histogram]
    total_hist = np.zeros(LEVELS)
    total_weights = 0
    total_bits = 0.0
    zero_rows = 0
    other = {}                   # dtype code -> [tensors, values]
    for name, (code, array) in tensors.items():
        a = np.asarray(array)
        if code != DTYPE_INT8:
            other.setdefault(code, [0, 0])
            other[code][0] += 1
            other[code][1] += a.size
            continue
        v = a.astype(np.int64)
        if v.ndim >= 2:
            zero_rows += int((v.reshape(v.shape[0], -1) == 0).all(axis=1).sum())
        hist = np.bincount(v.ravel() + 7, minlength=LEVELS).astype(np.float64)
        bits = entropy_bits(hist) * v.size
        f = families.setdefault(family(name), [0, 0.0, np.zeros(LEVELS)])
        f[0] += v.size
        f[1] += bits
        f[2] += hist
        total_hist += hist
        total_weights += v.size
        total_bits += bits

    print("%-52s %10s %8s %7s %7s" % ("family", "weights", "bits/w", "P(+-7)",
                                       "P(0)"))
    for g, (n, bits, hist) in sorted(families.items(), key=lambda x: -x[1][0]):
        p = hist / hist.sum()
        print("%-52s %10d %8.4f %7.4f %7.4f" % (g, n, bits / n, p[0] + p[-1],
                                                 p[7]))

    p = total_hist / total_hist.sum()
    global_bits = entropy_bits(total_hist) * total_weights
    size = os.path.getsize(path)
    print()
    print("int4 weights            %12d" % total_weights)
    print("histogram -7..7         " + " ".join("%.4f" % x for x in p))
    print("clip levels +-7         %12.4f" % (p[0] + p[-1]))
    print("level 0                 %12.4f" % p[7])
    print("all-zero rows           %12d" % zero_rows)
    print("per-tensor order 0      %12.0f B  (%.4f bits per weight)"
          % (total_bits / 8, total_bits / total_weights))
    print("one global histogram    %12.0f B  (%.4f bits per weight)"
          % (global_bits / 8, global_bits / total_weights))
    print("per-family trees save   %12.0f B at most"
          % ((global_bits - total_bits) / 8))
    print("container               %12d B" % size)
    print("rest of the container   %12.0f B  (scales, f32 tensors, names, flags)"
          % (size - total_bits / 8))
    for code, (n, values) in sorted(other.items()):
        print("dtype code %d            %12d tensors, %d values" % (code, n, values))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
