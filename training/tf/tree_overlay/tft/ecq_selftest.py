"""CPU self-test of tft/ecq.py (no GPU, no data)

    PYTHONPATH=<tree> python -m tft.ecq_selftest

Checks, on small random weight quantizers built exactly as CastedLinear builds
them:
  1. lambda = 0: forward and both gradients identical to his QuantizeFunction;
  2. lambda > 0: only floor/ceil moves inside [-7, 7], clipped weights as
     before, the rate goes down and the distortion up;
  3. the histogram moves once per optimizer step (AdamW and Muon bump the
     version counter of the weights; repeated forward passes count once);
  4. bake: round to nearest on the baked weights gives the chosen levels;
  5. the rate term: with no loss gradient the scale gradient is exactly
     -c * d_in / (s ln 2) under his LSQ normalisation.
"""

from __future__ import annotations

import math

import torch
from torch.nn import Module, Parameter

from pysrc.quantization import Quantization, Quantize, QuantizeFunction

from tft import ecq


def weight_quantization() -> Quantization:
    # tft/config.py make_quantizations(), weight part
    return Quantization(buckets=15, block_size=None, scale_dtype=torch.bfloat16,
                        arithmetic_dtype=torch.float32, scale_init="scaled_max",
                        epsilon=None)


class Owner(Module):
    def __init__(self, d_out: int, d_in: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = Parameter(torch.randn(d_out, d_in, generator=g) * 0.05)
        self.quantize_weight = Quantize(weight_quantization(), size=(d_out, d_in),
                                        batched=False, scale_weight_dtype=torch.float32,
                                        flatten_scale=True)
        with torch.no_grad():
            # a little below max|w|/7, so that some weights clip
            self.quantize_weight.scale.copy_(self.weight.abs().amax(1) / 6.0)

    def forward(self) -> torch.Tensor:
        return self.quantize_weight(self.weight, initialize_scales_steps=None,
                                    return_ints=False)


class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = Owner(48, 64, 1)
        self.b = Owner(64, 32, 2)


def reference(owner: Owner, grad: torch.Tensor):
    w = owner.weight.detach().clone().requires_grad_(True)
    s = owner.quantize_weight.scale.detach().clone().requires_grad_(True)
    blocks = owner.quantize_weight.reshape_input(w)
    out = QuantizeFunction.apply(blocks, s.view(-1), owner.quantize_weight.quant, False)
    out.view(w.shape).backward(grad)
    ints = QuantizeFunction.apply(blocks.detach(), s.detach().view(-1),
                                  owner.quantize_weight.quant, True)
    return out.detach().view(w.shape), w.grad, s.grad, ints.view(w.shape)


def check_lambda_zero() -> None:
    m = Model()
    m.train()
    grads = {k: torch.randn_like(o.weight) for k, o in (("a", m.a), ("b", m.b))}
    refs = {k: reference(o, grads[k]) for k, o in (("a", m.a), ("b", m.b))}
    ecq.install(m, 0.0, verbose=False)
    for k, o in (("a", m.a), ("b", m.b)):
        out = o()
        out.backward(grads[k])
        r_out, r_gw, r_gs, r_ints = refs[k]
        assert torch.equal(out.detach(), r_out), "lambda 0: forward differs"
        assert torch.equal(o.weight.grad, r_gw), "lambda 0: weight gradient differs"
        assert torch.equal(o.quantize_weight.scale.grad, r_gs), "lambda 0: scale gradient differs"
        assert torch.equal(ecq.levels_of(o, o.quantize_weight).view(r_ints.shape),
                           r_ints.to(torch.float32)), "lambda 0: levels differ"
    print("1. lambda 0 equals his QuantizeFunction: forward, gradients, levels")


def check_lambda_positive() -> None:
    m0 = Model()
    bits0, n = ecq.rate(m0)
    base = {k: ecq.levels_of(o, o.quantize_weight) for k, o in (("a", m0.a), ("b", m0.b))}
    m = Model()
    ecq.install(m, 0.3, decay=0.5, verbose=False)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)    # lr 0: weights fixed, versions move
    for _ in range(6):
        opt.zero_grad()
        (m.a().sum() + m.b().sum()).backward()
        opt.step()
        with torch.no_grad():
            m.a.weight.add_(0.0)                       # a new version, same values
            m.b.weight.add_(0.0)
    bits1, _ = ecq.rate(m)
    for k, o in (("a", m.a), ("b", m.b)):
        lv = ecq.levels_of(o, o.quantize_weight)
        scaled = (o.weight.detach()
                  / o.quantize_weight.scale.detach().to(torch.bfloat16).float()[:, None])
        scaled = scaled.view(lv.shape)
        inside = (scaled >= -7) & (scaled <= 7)
        assert torch.all((lv[inside] == scaled[inside].floor())
                         | (lv[inside] == scaled[inside].floor() + 1)), "not floor/ceil"
        assert torch.equal(lv[~inside], base[k][~inside]), "clipped weights moved"
    d0 = d1 = 0.0
    for k, o in (("a", m.a), ("b", m.b)):
        scaled = (o.weight.detach()
                  / o.quantize_weight.scale.detach().to(torch.bfloat16).float()[:, None]).view(-1)
        d0 += float((scaled - base[k].view(-1)).square().sum())
        d1 += float((scaled - ecq.levels_of(o, o.quantize_weight).view(-1)).square().sum())
    assert bits1 < bits0 and d1 > d0, (bits0, bits1, d0, d1)
    print("2. lambda 0.3: %.4f -> %.4f bits/weight, distortion %.4f -> %.4f steps^2"
          % (bits0 / n, bits1 / n, d0 / n, d1 / n))


def check_versions() -> None:
    for name, make in (("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3)),
                       ("Muon", lambda p: torch.optim.Muon(p, lr=1e-3))):
        m = Model()
        ecq.install(m, 0.1, decay=0.5, verbose=False)
        q = m.a.quantize_weight
        params = [m.a.weight, m.b.weight] if name == "Muon" else list(m.parameters())
        opt = make(params)
        m.a()                                    # first version: counts once
        m.a()
        m.a()                                    # same version: no further count
        assert int(q.ecq_pending.sum()) == m.a.weight.numel(), "counted more than once"
        h_before = q.ecq_hist.clone()
        v = m.a.weight._version
        (m.a().sum() + m.b().sum()).backward()
        opt.step()
        assert m.a.weight._version != v, name + " does not bump the version counter"
        m.a()                                    # new version: fold, count again
        assert not torch.equal(q.ecq_hist, h_before), "histogram did not move"
        assert int(q.ecq_pending.sum()) == m.a.weight.numel()
    print("3. AdamW and Muon bump the version; one count per version")


def check_bake() -> None:
    m = Model()
    ecq.install(m, 0.5, decay=0.5, verbose=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for _ in range(4):
        opt.zero_grad()
        (m.a().square().sum() + m.b().square().sum()).backward()
        opt.step()
    m.a()
    m.b()
    want = {k: ecq.levels_of(o, o.quantize_weight) for k, o in (("a", m.a), ("b", m.b))}
    moved = ecq.bake(m, verbose=False)
    for k, o in (("a", m.a), ("b", m.b)):
        q = o.quantize_weight
        ints = QuantizeFunction.apply(q.reshape_input(o.weight.detach()),
                                      q.scale.detach().view(-1), q.quant, True)
        assert torch.equal(ints.to(torch.float32), want[k]), "bake: levels not kept"
    print("4. bake: round to nearest gives the chosen levels back (%d weights moved)"
          % moved)


def check_rate_term() -> None:
    m = Model()
    c = 1e-3
    ecq.install(m, 0.0, verbose=False, rate_coeff=c)
    o = m.a
    o.train()
    out = o()
    out.backward(torch.zeros_like(out))          # no loss gradient at all
    q = o.quantize_weight
    s = q.scale.detach().to(torch.bfloat16).float()
    d_out, d_in = o.weight.shape
    want = -c * d_in / (s * math.log(2.0)) / math.sqrt(d_out * d_in * q.quant.qmax)
    assert torch.allclose(q.scale.grad, want, rtol=1e-6, atol=0.0), "rate gradient"
    assert bool((q.scale.grad < 0).all()), "the rate term must push the scales up"
    print("5. rate term: scale gradient -c d_in / (s ln 2), LSQ normalised")


def main() -> None:
    torch.manual_seed(0)
    check_lambda_zero()
    check_lambda_positive()
    check_versions()
    check_bake()
    check_rate_term()
    print("ecq self-test passed")


if __name__ == "__main__":
    main()
