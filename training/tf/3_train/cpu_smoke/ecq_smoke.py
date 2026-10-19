#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""CPU smoke test of the qat phase with ECQ on the shipped transformer

    PYTHONPATH=tf/3_train/cpu_smoke/fla_cpu:<trainer tree> \\
      python3 tf/3_train/cpu_smoke/ecq_smoke.py <checkpoint.tch> <shipped.bin> \\
      [--dump <tfdump.bin>] [--tokens 128] [--steps 2] [--out <dir>]

<trainer tree> is tree_upstream with tree_overlay copied over it (as on the
GPU host); fla_cpu/ stands in for the GPU kernels of fla with the pure-torch
references of pysrc/kda_reference.py, and the attention runs as sdpa.
<checkpoint.tch> and <shipped.bin> come from tf/4_export/container_to_tch.py
(--tch, --bin) on the shipped container.

What it runs, as tft/runner.py does for the qat phase of arm pkd: the qat
model with the context package, the checkpoint loaded (no missing or
unexpected keys), ECQ installed from TFT_ECQ_* (defaults lambda 0.05,
mu 6.1e-9), then `--steps` AdamW steps on one micro batch, ecq.bake, a load
into the fp32 model (strict) and tft/export_weights_poc.py.  Checks: finite
gradients, a gradient on every quantizer scale, the frozen all-zero rows
unchanged, and the export against <shipped.bin> (which integer entries
moved).  The batch is `--tokens` consecutive stream bytes of a transformer
dump of a development run with its PPMd rows as priors (dev/README.md), or,
without --dump, fixed pseudo-random tokens with flat priors.  No prompt rows
and no article context: the loss is not comparable with the coder's.

A few seconds per step for 128 tokens; it ends with "SMOKE OK".
"""

import argparse
import dataclasses
import os
import sys
import time

V = 208


def batch(args):
    import numpy as np
    import torch

    from zctx import zpaket

    n = args.tokens + 1
    if args.dump:
        rec_t = np.dtype([("pos", "<u4"), ("byte", "u1"), ("zone", "u1"), ("pad", "<u2"),
                          ("slot", "<f2", (V,)), ("hidden", "<f2", (192,)),
                          ("ppmd", "<f2", (V,))])
        set_bytes = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12] + list(range(32, 58)) + [64]
                             + list(range(74, 84)) + [88] + list(range(91, 96))
                             + list(range(97, 123)) + list(range(128, 256)))
        index_of = np.full(256, -1)
        index_of[set_bytes] = np.arange(V)
        rec = np.asarray(np.memmap(args.dump, dtype=rec_t, mode="r")[args.skip:args.skip + n])
        tok = index_of[rec["byte"]]
        assert (tok >= 0).all()
        pri = rec["ppmd"][1:].astype(np.float32)
    else:
        tok = np.random.default_rng(7).integers(0, V, n)
        pri = np.ones((n - 1, V), np.float32)
    pri = (pri / pri.sum(1, keepdims=True)).astype(np.float16)
    t = args.tokens
    # the package carrier: token, its predecessors, no match (mark 208, class 0)
    inp = torch.from_numpy(zpaket.pack(tok[:-1], np.full(t, 208), np.zeros(t, np.int64)))
    return dict(input_tokens=inp[None],
                priors=torch.from_numpy(pri)[None],
                loss_mask=torch.ones(1, t, dtype=torch.bool),
                output_tokens=torch.from_numpy(tok[1:].astype(np.int64))[None],
                cumulative_sequence_lengths=torch.tensor([[0, t]], dtype=torch.int32))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tch")
    ap.add_argument("shipped_bin")
    ap.add_argument("--dump")
    ap.add_argument("--skip", type=int, default=100000, help="first dump record used")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default="cpu_smoke_out")
    args = ap.parse_args(argv[1:])
    os.environ.update(TFT_GEOMETRY="poc8m_w2048", ZCTX_ARM="pkd", PK_LAST_X0ROW="1",
                      PK_SLOT_DIM="16", PK_LAST_X="2", PK_REDIRMERGE="1")
    os.environ.setdefault("TFT_ECQ_LAMBDA", "0.05")
    os.environ.setdefault("TFT_ECQ_RATE_MU", "6.1e-9")

    import numpy as np
    import torch

    from pysrc.export_weights import read_tensor_file
    from pysrc.model import Transformer
    from tft import config as tftcfg
    from tft import ecq
    from tft import export_weights_poc as ewp
    from zctx.zpmodel import install_paket

    torch.manual_seed(0)
    # subnormal decays of the KDA reference make real data ~40x slower on CPU
    torch.set_flush_denormal(True)
    g = tftcfg.geometry()

    def build(phase):
        cfg = tftcfg.make_config(phase, geometry_name=g.name, variable_sequence_length=True)
        m = Transformer(dataclasses.replace(cfg, attention_implementation="sdpa"))
        install_paket(m, cfg, x=2, d_slot=16, use_last=True, use_match=True)
        return m

    model = build("qat")
    res = model.load_state_dict(torch.load(args.tch, map_location="cpu"), strict=False)
    assert not res.missing_keys and not res.unexpected_keys, res
    print("qat model: checkpoint loaded, no missing or unexpected keys")
    assert ecq.install_from_env(model, loss_tokens=args.tokens, accumulation=1)
    frozen = []
    for _name, owner, q in ecq.weight_quantizers(model):
        rows = (q.reshape_input(owner.weight) == 0).all(-1).view(-1)
        frozen.append((owner, q, rows, owner.weight.detach().clone(), q.scale.detach().clone()))
    b = batch(args)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2.5e-6, weight_decay=0.0)
    model.train()
    for step in range(args.steps):
        t0 = time.time()
        loss = model(**b)
        loss.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        nograd = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        scales = [n for n, p in model.named_parameters()
                  if n.endswith(".scale") and p.grad is not None and bool(p.grad.any())]
        print("step %d: loss %.4f nat/token, %.0f s; %d non-finite gradients, %d parameters "
              "without gradient, %d scales with a gradient"
              % (step, loss.item() / args.tokens, time.time() - t0, len(bad), len(nograd),
                 len(scales)), flush=True)
        assert not bad and not nograd, (bad, nograd)
        opt.step()
        opt.zero_grad()
    moved_rows = 0
    for owner, q, rows, w0, s0 in frozen:
        w = q.reshape_input(owner.weight.detach())[0]
        w_before = q.reshape_input(w0)[0]
        moved_rows += int((w[rows] != w_before[rows]).any(-1).sum())
        moved_rows += int((q.scale.detach().view(-1)[rows] != s0.view(-1)[rows]).sum())
    print("frozen all-zero rows: %d rows, %d changed"
          % (sum(int(f[2].sum()) for f in frozen), moved_rows))
    assert moved_rows == 0, "a frozen row moved"
    ecq.bake(model)
    fp = build("fp32")
    fp.load_state_dict(model.state_dict(), strict=True)
    fp.eval()
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "smoke_export.bin")
    ewp.export_weights(fp, dst, g.name, ewp.paket_from_env())
    a = read_tensor_file(args.shipped_bin)
    c = read_tensor_file(dst)
    assert list(a) == list(c), "tensor order differs from the shipped file"
    total = changed = 0
    for n in a:
        if a[n][1].dtype.kind in "iu":
            total += a[n][1].size
            d = a[n][1] != c[n][1]
            if d.any():
                changed += int(d.sum())
                print("  moved: %-62s %d entries" % (n, int(d.sum())))
    print("export: %d of %d integer entries differ from the shipped file" % (changed, total))
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
