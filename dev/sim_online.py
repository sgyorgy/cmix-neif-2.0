#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""offline simulation of online adaptation on the transformer state
(PLAN_10PCT.md phase 2.1).  NOT part of the submission (dev/README.md).

  python3 dev/sim_online.py <tfdump.bin> <cost.u16> [--arms adapter,lsh]

Input: a transformer dump of a development run (NEIF_DEV_TFDUMP, see
dev/tfdump.py) and the cost file of the same run (NEIF_DEV_COST).  Gated
records (zone != 0) are skipped: there the coder does not read the
transformer.

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

Deterministic (fixed seeds).  Needs numpy; torch is used when importable
(much faster), else numpy.
"""

import argparse
import math
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
    lo = np.zeros(n, dtype=np.int64)          # byte range [lo, lo + width)
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


def coder_bit_probs(cost_bits, bits):
    """the coder's P(bit = 1) from the cost of the coded bit (bits)"""
    p_act = np.power(2.0, -cost_bits)
    return np.clip(np.where(bits == 1, p_act, 1 - p_act), 1e-6, 1 - 1e-6)


def mixer_cost(inputs, bits, lr=0.02):
    """online logistic mixer per bit position: inputs (n, 8, k) stretched,
    returns the total cost in bits"""
    n, _, k = inputs.shape
    w = np.zeros((8, k))
    w[:, 0] = 1.0                               # start as the coder alone
    total = 0.0
    for t in range(n):
        x = inputs[t]                            # (8, k)
        z = np.einsum("jk,jk->j", w, x)
        p = 1 / (1 + np.exp(-z))
        y = bits[t]
        pa = np.where(y == 1, p, 1 - p)
        total -= np.log2(np.clip(pa, 1e-12, 1)).sum()
        w += lr * (y - p)[:, None] * x
    return total


def run_adapter(r, idx, lr, use_torch):
    n = len(idx)
    logp = np.log(np.clip(r["slot"].astype(np.float64), EPS, None))
    h = r["hidden"].astype(np.float64)
    out = np.empty((n, V))
    if use_torch:
        import torch
        W = torch.zeros(V, D, dtype=torch.float64)
        b = torch.zeros(V, dtype=torch.float64)
        LP = torch.from_numpy(logp)
        H = torch.from_numpy(h)
        for t in range(n):
            z = LP[t] + W @ H[t] + b
            p = torch.softmax(z, 0)
            out[t] = p.numpy()
            g = p.clone()
            g[idx[t]] -= 1.0                     # d CE / d z
            W -= lr * torch.outer(g, H[t])
            b -= lr * g
    else:
        W = np.zeros((V, D))
        b = np.zeros(V)
        for t in range(n):
            z = logp[t] + W @ h[t] + b
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            out[t] = p
            g = p.copy()
            g[idx[t]] -= 1.0
            W -= lr * np.outer(g, h[t])
            b -= lr * g
    return out


def run_lsh(r, idx, k_bits, seed=7, lr=0.002):
    n = len(idx)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((D, k_bits))
    h = r["hidden"].astype(np.float64)
    sig = (h @ proj > 0).astype(np.int64)
    code = (sig * (1 << np.arange(k_bits))).sum(axis=1)
    prev = np.concatenate([[0], idx[:-1]])
    ctx = code * V + prev
    counts = {}
    p_tf = np.clip(r["slot"].astype(np.float64), EPS, None)
    p_tf /= p_tf.sum(axis=1, keepdims=True)
    out = np.empty((n, V))
    w = 0.0                                      # weight of the lsh model
    for t in range(n):
        c = counts.get(ctx[t])
        if c is None:
            q = np.full(V, 1.0 / V)
        else:
            q = (c + 0.02) / (c.sum() + 0.02 * V)
        z = np.log(p_tf[t]) + w * (np.log(q) - np.log(p_tf[t]))
        z -= z.max()
        m = np.exp(z)
        m /= m.sum()
        out[t] = m
        # gradient of -log m[y] w.r.t. w
        d = np.log(q) - np.log(p_tf[t])
        g = -(d[idx[t]] - (m * d).sum())
        w = min(1.0, max(0.0, w - lr * g))
        if c is None:
            c = np.zeros(V)
            counts[ctx[t]] = c
        c[idx[t]] += 1.0
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("cost")
    ap.add_argument("--arms", default="adapter,lsh")
    ap.add_argument("--limit", type=int, default=0, help="first N records only")
    ap.add_argument("--adapter-lr", type=float, default=0.002)
    ap.add_argument("--lsh-bits", type=int, default=16)
    args = ap.parse_args(argv[1:])
    r = np.memmap(args.dump, dtype=RECORD, mode="r")
    if args.limit:
        r = r[:args.limit]
    keep = np.flatnonzero(r["zone"] == 0)
    r = np.asarray(r[keep])
    idx = INDEX_OF[r["byte"]]
    assert (idx >= 0).all()
    n = len(idx)
    c = np.fromfile(args.cost, dtype="<u2").reshape(-1, 8)
    cost_bits = c[r["pos"].astype(np.int64)].astype(np.float64) / 2048.0
    print("%d records outside the gated zones" % n)
    try:
        import torch  # noqa: F401
        use_torch = True
    except ImportError:
        use_torch = False

    p_tf = np.clip(r["slot"].astype(np.float64), EPS, None)
    p_tf /= p_tf.sum(axis=1, keepdims=True)
    ce = lambda dist: float(-np.log2(dist[np.arange(n), idx]).mean())
    print("transformer slot      %.4f bits/B" % ce(p_tf))
    print("coder                 %.4f bits/B" % float(cost_bits.sum(axis=1).mean()))

    tf_bits, bits = bit_probs(p_tf, idx)
    pc = coder_bit_probs(cost_bits, bits)
    base_in = np.stack([stretch(pc), stretch(tf_bits), np.ones_like(pc)], axis=2)
    ref = mixer_cost(base_in, bits) / n
    alone = mixer_cost(np.stack([stretch(pc), np.ones_like(pc)], axis=2), bits) / n
    print("mixer coder alone     %.4f bits/B   coder + slot %.4f bits/B" % (alone, ref))
    for arm in args.arms.split(","):
        if arm == "adapter":
            dist = run_adapter(r, idx, args.adapter_lr, use_torch)
            name = "adapter lr %g" % args.adapter_lr
        elif arm == "lsh":
            dist = run_lsh(r, idx, args.lsh_bits)
            name = "lsh %d bits" % args.lsh_bits
        else:
            sys.exit("unknown arm " + arm)
        ab, _ = bit_probs(dist, idx)
        m = mixer_cost(np.stack([stretch(pc), stretch(ab), np.ones_like(pc)],
                                axis=2), bits) / n
        print("%-20s  %.4f bits/B (slot %+.4f); coder proxy %.4f (%+.4f against "
              "coder + slot)" % (name, ce(dist), ce(dist) - ce(p_tf), m, m - ref))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
