#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""offline tests of the coder's final stage (PLAN_10PCT.md 1.2, section 10).
NOT part of the submission (dev/README.md).

  python3 dev/sim_final.py <tfdump.bin> <cost.u16> [--tests apm,remix,slot]

Input as for dev/sim_online.py: a transformer dump and the cost file of the
same run.  The coder's probability of each coded bit comes from its cost (the
real cost, 1/2048 bit); the transformer's and PPMd's bit probabilities from
their rows in the dump.  Gated records (zone != 0) are skipped.  Every model
runs causally (predict, then update) in stream order.

  apm    an APM behind the coder (33 interpolation points over stretch(p),
         rate 1/(n + 1.5) down to 1/limit), keyed by the partial byte and by
         buckets of the transformer state: its bit probability, the
         probability of its byte-start favourite while that byte is still
         possible, its top probability among the still possible bytes; and,
         as a control, by the previous byte.  Output alone, (p + 3 apm) / 4
         and an online 2-input mixer.
  remix  an online logistic mixer over stretch(p_coder) and stretch(p_ppmd)
         and/or stretch(p_tf), weight sets selected by bit position and by
         confidence buckets of PPMd and of the transformer.
  slot   cross entropy of the slot under a temperature, and of a linear mix
         of the slot with the PPMd row.

Each line gives the difference to the coder in bits per byte (negative =
better than the coder).  Needs numpy and numba.
"""

import argparse
import sys

import numpy as np
from numba import njit

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
QTHR = np.array([.20, .40, .60, .75, .85, .92, .97])   # src/tf_sharpen.h
PMIN = 1.0 / 65536


def norm_rows(rows):
    p = np.clip(rows.astype(np.float64), 1e-7, None)
    return p / p.sum(axis=1, keepdims=True)


def bit_features(dist, byte):
    """per coded bit: P(1 | bits so far), the bucket of the top probability
    among the still possible bytes, the bucket of the byte-start favourite
    (8 = no longer possible), the bit, the partial byte"""
    n = len(byte)
    full = np.zeros((n, 256))
    full[:, SET_BYTES] = dist
    top = full.argmax(axis=1)
    qbest = np.searchsorted(QTHR, full.max(axis=1))
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(full, axis=1)], axis=1)
    rows = np.arange(n)
    p1 = np.empty((n, 8))
    qcond = np.empty((n, 8), np.int64)
    qtop = np.empty((n, 8), np.int64)
    y = np.empty((n, 8), np.int8)
    c0 = np.empty((n, 8), np.int64)
    lo = np.zeros(n, np.int64)
    for j in range(8):
        w = 256 >> j
        mid = lo + w // 2
        p_all = csum[rows, lo + w] - csum[rows, lo]
        p1[:, j] = np.where(p_all > 0, (csum[rows, lo + w] - csum[rows, mid])
                            / np.maximum(p_all, 1e-300), 0.5)
        blk = full.reshape(n, 1 << j, w).max(axis=2)
        qcond[:, j] = np.searchsorted(QTHR, blk[rows, lo >> (8 - j)]
                                      / np.maximum(p_all, 1e-300))
        qtop[:, j] = np.where((top >> (8 - j)) == (lo >> (8 - j)), qbest, 8)
        c0[:, j] = (1 << j) | (lo >> (8 - j))
        b = (byte >> (7 - j)) & 1
        y[:, j] = b
        lo = np.where(b == 1, mid, lo)
    return np.clip(p1, PMIN, 1 - PMIN), qcond, qtop, y, c0


@njit(cache=False)
def _apm(st, ctx, y, nctx, limit, out):
    t = np.empty(nctx * 33)
    for c in range(nctx):
        for k in range(33):
            t[c * 33 + k] = 1.0 / (1.0 + np.exp(-(k - 16) * 0.5))
    cnt = np.zeros(nctx * 33)
    for i in range(len(st)):
        s = min(max(st[i], -7.999), 7.999)
        pos = (s + 8.0) * 2.0
        lo = int(pos)
        w = pos - lo
        a = ctx[i] * 33 + lo
        out[i] = t[a] * (1.0 - w) + t[a + 1] * w
        t[a] += (y[i] - t[a]) * (1.0 - w) / (cnt[a] + 1.5)
        t[a + 1] += (y[i] - t[a + 1]) * w / (cnt[a + 1] + 1.5)
        cnt[a] = min(cnt[a] + 1.0 - w, limit)
        cnt[a + 1] = min(cnt[a + 1] + w, limit)


@njit(cache=False)
def _mix(inp, sel, y, nsel, lr):
    n, k = inp.shape
    w = np.zeros((nsel, k))
    w[:, 0] = 1.0
    total = 0.0
    for i in range(n):
        s = sel[i]
        z = 0.0
        for a in range(k):
            z += w[s, a] * inp[i, a]
        p = min(max(1.0 / (1.0 + np.exp(-z)), PMIN), 1.0 - PMIN)
        total -= np.log2(p if y[i] == 1 else 1.0 - p)
        g = lr * (y[i] - p)
        for a in range(k):
            w[s, a] += g * inp[i, a]
    return total


def stretch(p):
    return np.log(p / (1 - p))


def bits(p, y):
    return -np.log2(np.where(y == 1, p, 1 - p)).sum()


def load(args):
    rec = np.memmap(args.dump, dtype=RECORD, mode="r")
    if args.limit:
        rec = rec[:args.limit]
    cost = np.memmap(args.cost, dtype="<u2", mode="r").reshape(-1, 8)
    cols = {k: [] for k in ("y", "pc", "pt", "qct", "qtt", "c0", "pq", "qcq",
                            "byte", "slot_bits")}
    temps = (0.9, 0.95, 1.05, 1.1)
    mixes = (0.02, 0.05)
    slot = {t: 0.0 for t in temps}
    slot.update({("mix", a): 0.0 for a in mixes})
    for lo in range(0, len(rec), 50000):
        r = np.asarray(rec[lo:lo + 50000])
        r = r[r["zone"] == 0]
        if not len(r):
            continue
        byte = r["byte"].astype(np.int64)
        idx = INDEX_OF[byte]
        rows = np.arange(len(r))
        pt_rows, pq_rows = norm_rows(r["slot"]), norm_rows(r["ppmd"])
        pt, qct, qtt, y, c0 = bit_features(pt_rows, byte)
        pq, qcq, _, _, _ = bit_features(pq_rows, byte)
        cb = cost[r["pos"].astype(np.int64)].astype(np.float64) / 2048.0
        pa = np.power(2.0, -cb)
        pc = np.clip(np.where(y == 1, pa, 1.0 - pa), PMIN, 1 - PMIN)
        for k, v in (("y", y), ("pc", pc), ("pt", pt), ("qct", qct), ("qtt", qtt),
                     ("c0", c0), ("pq", pq), ("qcq", qcq)):
            cols[k].append(v.reshape(-1))
        cols["byte"].append(byte)
        lp = np.log(pt_rows)
        cols["slot_bits"].append(-np.log2(pt_rows[rows, idx]))
        for t in temps:
            z = lp / t
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            slot[t] -= np.log2(e[rows, idx] / e.sum(axis=1)).sum()
        for a in mixes:
            slot[("mix", a)] -= np.log2((1 - a) * pt_rows[rows, idx]
                                        + a * pq_rows[rows, idx]).sum()
    data = {k: np.ascontiguousarray(np.concatenate(v)) for k, v in cols.items()}
    return data, slot


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("cost")
    ap.add_argument("--tests", default="apm,remix,slot")
    ap.add_argument("--limit", type=int, default=0, help="first N records only")
    args = ap.parse_args(argv[1:])
    tests = args.tests.split(",")
    d, slot = load(args)
    y, pc = d["y"], d["pc"]
    n = len(d["byte"])
    base = bits(pc, y)
    print("%d bytes outside the gated zones: coder %.5f, transformer slot %.5f bits/B"
          % (n, base / n, d["slot_bits"].sum() / n))
    j = np.tile(np.arange(8), n)
    st_c = stretch(pc)
    if "slot" in tests:
        s1 = d["slot_bits"].sum()
        for key, v in slot.items():
            name = ("slot, temperature %.2f" % key if not isinstance(key, tuple)
                    else "slot + %.2f PPMd (linear)" % key[1])
            print("%-34s %+.5f bits/B against the plain slot" % (name, (v - s1) / n))
    if "apm" in tests:
        qtf = np.clip(((stretch(d["pt"]) + 8) * 0.75).astype(np.int64), 0, 11)
        c0 = d["c0"]
        ctxs = {      # built one at a time: each is 8 bytes per coded bit
            "c0": lambda: (c0, 256),
            "c0 x tf bit": lambda: (c0 * 12 + qtf, 256 * 12),
            "c0 x tf favourite": lambda: (c0 * 9 + d["qtt"], 256 * 9),
            "c0 x tf top": lambda: (c0 * 8 + d["qct"], 256 * 8),
            "c0 x tf bit x tf top": lambda: ((c0 * 12 + qtf) * 8 + d["qct"], 256 * 96),
            "c0 x previous byte": lambda: (c0 * 256 + np.repeat(
                np.concatenate([[0], d["byte"][:-1]]), 8), 256 * 256),
        }
        for name, make in ctxs.items():
            ctx, nctx = make()
            for limit in (127.0, 1023.0):
                out = np.empty(len(st_c))
                _apm(st_c, np.ascontiguousarray(ctx), y, nctx, limit, out)
                pa = np.clip(out, PMIN, 1 - PMIN)
                avg = bits(np.clip((pc + 3 * pa) / 4, PMIN, 1 - PMIN), y)
                mix = _mix(np.ascontiguousarray(np.stack([st_c, stretch(pa),
                                                          np.ones_like(pc)], 1)),
                           j, y, 8, 0.0005)
                print("apm %-22s limit %4d: alone %+.5f  (p+3apm)/4 %+.5f  mixer %+.5f"
                      % (name, limit, (bits(pa, y) - base) / n, (avg - base) / n,
                         (mix - base) / n), flush=True)
    if "remix" in tests:
        one = np.ones_like(pc)
        st_q, st_t = stretch(d["pq"]), stretch(d["pt"])
        sels = (("bit position", j, 8), ("x PPMd top", j * 8 + d["qcq"], 64),
                ("x PPMd top x tf top", (j * 8 + d["qcq"]) * 8 + d["qct"], 512))
        for name, cols in (("coder", [st_c, one]), ("coder + PPMd", [st_c, st_q, one]),
                           ("coder + tf", [st_c, st_t, one]),
                           ("coder + PPMd + tf", [st_c, st_q, st_t, one])):
            inp = np.ascontiguousarray(np.stack(cols, 1))
            for sname, sel, ns in sels:
                t = _mix(inp, np.ascontiguousarray(sel), y, ns, 0.0005)
                print("remix %-18s sets by %-20s %+.5f" % (name, sname, (t - base) / n),
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
