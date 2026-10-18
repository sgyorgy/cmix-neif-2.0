#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""offline simulation of online adaptation on the transformer state
(PLAN_10PCT.md phase 2.1).  NOT part of the submission (dev/README.md).

  python3 dev/sim_online.py <tfdump.bin> <cost.u16> [--arms adapter,lsh]

Input: a transformer dump of a development run (NEIF_DEV_TFDUMP, see
dev/tfdump.py) and the cost file of the same run (NEIF_DEV_COST).  Gated
records (zone != 0) are skipped: there the coder does not read the
transformer.  The records are read in chunks, so memory stays small.

Arms, each run causally (predict, then learn) over the records in order:
  adapter  softmax(log p_tf + W h + b) with W (208 x 192) and b learned
           online by SGD on the cross entropy of the coded byte;
  lsh      a byte-count model in the context (sign bits of K fixed random
           projections of h, previous byte), mixed with p_tf in the log
           domain with one online weight.

Measures (bits per byte, on the records outside the gated zones):
  * the cross entropy of the transformer slot itself and of each arm;
  * a proxy for the coder: an online logistic mixer per bit position over
    stretch(p_coder) (recovered from the coder's own per-bit cost) and
    stretch(p_x) for x = the plain slot (reference) or an arm; the arm's
    gain is its mixer's cost minus the reference mixer's cost.  The plan's
    go/no-go is about 0.0058 bits per byte (0.004 nat).

Deterministic (fixed seeds).  Needs numpy; with numba the loops are
compiled (minutes instead of hours for 2M records).
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
SET_BYTES = np.array(
    [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12] + list(range(32, 58)) + [64]
    + list(range(74, 84)) + [88] + list(range(91, 96)) + list(range(97, 123))
    + list(range(128, 256)), dtype=np.int64)
INDEX_OF = np.full(256, -1, dtype=np.int64)
INDEX_OF[SET_BYTES] = np.arange(V)
EPS = 1e-6

try:
    from numba import njit
except ImportError:          # plain Python loops, slow
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=False)
def _mixer_run(inputs, bits, w, lr):
    n, nb, k = inputs.shape
    total = 0.0
    for t in range(n):
        for j in range(nb):
            z = 0.0
            for i in range(k):
                z += w[j, i] * inputs[t, j, i]
            p = 1.0 / (1.0 + np.exp(-z))
            y = bits[t, j]
            pa = p if y == 1 else 1.0 - p
            if pa < 1e-12:
                pa = 1e-12
            total -= np.log2(pa)
            g = lr * (y - p)
            for i in range(k):
                w[j, i] += g * inputs[t, j, i]
    return total


@njit(cache=False)
def _adapter_run(logp, h, idx, W, b, lr, out):
    n, v = logp.shape
    d = h.shape[1]
    z = np.empty(v)
    for t in range(n):
        zmax = -1e300
        for a in range(v):
            s = logp[t, a] + b[a]
            for c in range(d):
                s += W[a, c] * h[t, c]
            z[a] = s
            if s > zmax:
                zmax = s
        tot = 0.0
        for a in range(v):
            z[a] = np.exp(z[a] - zmax)
            tot += z[a]
        for a in range(v):
            p = z[a] / tot
            out[t, a] = p
            g = p - (1.0 if a == idx[t] else 0.0)
            gl = lr * g
            b[a] -= gl
            for c in range(d):
                W[a, c] -= gl * h[t, c]


@njit(cache=False)
def _lsh_run(p_tf, code, idx, prev0, table, mask, w0, lr, out):
    n, v = p_tf.shape
    prev = prev0
    w = w0
    q = np.empty(v)
    dd = np.empty(v)
    m = np.empty(v)
    for t in range(n):
        slot = ((code[t] * 2654435761 + prev * 40503) & mask)
        tot = 0.0
        for a in range(v):
            tot += table[slot, a]
        for a in range(v):
            q[a] = (table[slot, a] + 0.02) / (tot + 0.02 * v)
        zmax = -1e300
        for a in range(v):
            lt = np.log(p_tf[t, a])
            dd[a] = np.log(q[a]) - lt
            m[a] = lt + w * dd[a]
            if m[a] > zmax:
                zmax = m[a]
        s = 0.0
        for a in range(v):
            m[a] = np.exp(m[a] - zmax)
            s += m[a]
        md = 0.0
        for a in range(v):
            m[a] /= s
            out[t, a] = m[a]
            md += m[a] * dd[a]
        g = -(dd[idx[t]] - md)
        w -= lr * g
        if w < 0.0:
            w = 0.0
        if w > 1.0:
            w = 1.0
        table[slot, idx[t]] += 1.0
        prev = idx[t]
    return prev, w


def stretch(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bit_probs(dist, idx):
    """P(bit = 1 | bits so far) for the 8 bits of each coded byte, from a
    distribution over the set (rows sum to 1); returns (n, 8) and the bits"""
    n = len(idx)
    full = np.zeros((n, 256))
    full[:, SET_BYTES] = dist
    byte = SET_BYTES[idx]
    out = np.empty((n, 8))
    bits = np.empty((n, 8), dtype=np.int64)
    lo = np.zeros(n, dtype=np.int64)
    width = 256
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(full, axis=1)], axis=1)
    rows = np.arange(n)
    for j in range(8):
        width //= 2
        mid = lo + width
        p_hi = csum[rows, mid + width] - csum[rows, mid]
        p_all = csum[rows, mid + width] - csum[rows, lo]
        out[:, j] = np.where(p_all > 0, p_hi / np.maximum(p_all, 1e-300), 0.5)
        b = (byte >> (7 - j)) & 1
        bits[:, j] = b
        lo = np.where(b == 1, mid, lo)
    return np.clip(out, 1e-6, 1 - 1e-6), bits


class BitMixer:
    """online logistic mixer per bit position; input 0 starts at weight 1"""

    def __init__(self, k, lr=0.02):
        self.w = np.zeros((8, k))
        self.w[:, 0] = 1.0
        self.lr = lr
        self.bits = 0.0

    def run(self, inputs, bits):
        self.bits += _mixer_run(np.ascontiguousarray(inputs), np.ascontiguousarray(bits),
                                self.w, self.lr)


class Adapter:
    def __init__(self, lr):
        self.lr = lr
        self.W = np.zeros((V, D))
        self.b = np.zeros(V)

    def run(self, logp, h, idx):
        out = np.empty((len(idx), V))
        _adapter_run(np.ascontiguousarray(logp), np.ascontiguousarray(h), idx,
                     self.W, self.b, self.lr, out)
        return out


class Lsh:
    def __init__(self, k_bits, seed=7, lr=0.002, table_bits=18):
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((D, k_bits))
        self.k = k_bits
        self.table = np.zeros((1 << table_bits, V), dtype=np.float32)
        self.mask = (1 << table_bits) - 1
        self.w = 0.0
        self.lr = lr
        self.prev = 0

    def run(self, p_tf, h, idx):
        sig = (h @ self.proj > 0).astype(np.int64)
        code = (sig * (1 << np.arange(self.k))).sum(axis=1)
        out = np.empty((len(idx), V))
        self.prev, self.w = _lsh_run(np.ascontiguousarray(p_tf), code, idx, self.prev,
                                     self.table, self.mask, self.w, self.lr, out)
        return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("cost")
    ap.add_argument("--arms", default="adapter,lsh")
    ap.add_argument("--limit", type=int, default=0, help="first N records only")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--adapter-lr", type=float, default=0.002)
    ap.add_argument("--lsh-bits", type=int, default=16)
    ap.add_argument("--mix-lr", type=float, default=0.002)
    args = ap.parse_args(argv[1:])
    rec = np.memmap(args.dump, dtype=RECORD, mode="r")
    if args.limit:
        rec = rec[:args.limit]
    cost = np.memmap(args.cost, dtype="<u2", mode="r").reshape(-1, 8)
    arms = [a for a in args.arms.split(",") if a]
    adapter = Adapter(args.adapter_lr) if "adapter" in arms else None
    lsh = Lsh(args.lsh_bits) if "lsh" in arms else None
    m_alone = BitMixer(2, args.mix_lr)
    m_ref = BitMixer(3, args.mix_lr)
    m_arm = {a: BitMixer(3, args.mix_lr) for a in arms}
    ce = {"slot": 0.0, "coder": 0.0}
    ce.update({a: 0.0 for a in arms})
    n_total = 0
    for lo in range(0, len(rec), args.chunk):
        r = np.asarray(rec[lo:lo + args.chunk])
        r = r[r["zone"] == 0]
        if len(r) == 0:
            continue
        idx = INDEX_OF[r["byte"]]
        assert (idx >= 0).all()
        n = len(idx)
        n_total += n
        rows = np.arange(n)
        cbits = cost[r["pos"].astype(np.int64)].astype(np.float64) / 2048.0
        p_tf = np.clip(r["slot"].astype(np.float64), EPS, None)
        p_tf /= p_tf.sum(axis=1, keepdims=True)
        h = r["hidden"].astype(np.float64)
        ce["slot"] -= np.log2(p_tf[rows, idx]).sum()
        ce["coder"] += cbits.sum()
        tf_bits, bits = bit_probs(p_tf, idx)
        p_act = np.power(2.0, -cbits)
        pc = np.clip(np.where(bits == 1, p_act, 1 - p_act), 1e-6, 1 - 1e-6)
        one = np.ones_like(pc)
        m_alone.run(np.stack([stretch(pc), one], axis=2), bits)
        m_ref.run(np.stack([stretch(pc), stretch(tf_bits), one], axis=2), bits)
        for a in arms:
            if a == "adapter":
                dist = adapter.run(np.log(p_tf), h, idx)
            elif a == "lsh":
                dist = lsh.run(p_tf, h, idx)
            else:
                sys.exit("unknown arm " + a)
            ce[a] -= np.log2(np.clip(dist[rows, idx], 1e-12, 1)).sum()
            ab, _ = bit_probs(dist, idx)
            m_arm[a].run(np.stack([stretch(pc), stretch(ab), one], axis=2), bits)
        print("... %d records" % n_total, file=sys.stderr, flush=True)
    n = max(n_total, 1)
    print("%d records outside the gated zones" % n_total)
    print("transformer slot      %.4f bits/B" % (ce["slot"] / n))
    print("coder                 %.4f bits/B" % (ce["coder"] / n))
    print("mixer coder alone     %.4f bits/B   coder + slot %.4f bits/B"
          % (m_alone.bits / n, m_ref.bits / n))
    for a in arms:
        name = ("adapter lr %g" % args.adapter_lr if a == "adapter"
                else "lsh %d bits" % args.lsh_bits)
        print("%-20s  %.4f bits/B (slot %+.4f); coder proxy %.4f (%+.4f against "
              "coder + slot)" % (name, ce[a] / n, (ce[a] - ce["slot"]) / n,
                                 m_arm[a].bits / n, (m_arm[a].bits - m_ref.bits) / n))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
