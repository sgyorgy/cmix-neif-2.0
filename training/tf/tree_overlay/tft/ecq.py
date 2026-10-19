"""entropy-constrained rounding of the int4 weights during QAT (ECQ)

Derived from Vladimir Ivanov's pysrc/quantization.py (fx2-cmix-transformer,
GPL-3): the forward arithmetic (bf16 row scale, fp32 division, clamp to
[-7, 7]) and the backward pass are his QuantizeFunction, with the chosen level
in place of round(w/s); what this module adds is the choice of the level, the
rate term and the bake.

PLAN_10PCT.md phase 3.  The container codes the int4 levels of every weight
tensor with an adaptive order-0 model (weights_compress_v3.py), so a weight
costs about -log2 P(level) bits, 3.73 bits on average in the shipped
container.  His quantizer (pysrc/quantization.py, QuantizeFunction) rounds
every weight to the NEAREST level.  ECQ instead picks, between the two levels
around w/s, the one that minimises

    (w/s - k)^2 + lambda * (-log2 P_k)

with P the running level histogram of the tensor: where moving a weight to
the cheaper neighbour costs little distortion, it moves.  lambda is in squared
quantization steps per bit; lambda = 0 is round to nearest (ties to even, as
torch.round).  Weights outside [-7, 7] are clamped as before, so the gradient
of the row scale keeps his LSQ form; in the backward pass the chosen level
takes the place of round(w/s).

How it is wired (no file of his is changed):

  * install(model, lam) swaps the class of every 15-level weight quantizer
    (CastedLinear, CastedEmbedding: `<module>.quantize_weight`) to
    ECQQuantize.  The state_dict keys stay the same, the histograms are
    non-persistent buffers.
  * The histogram moves once per optimizer step: a quantizer counts the levels
    of the first forward pass that sees a new version of its weight tensor
    (the optimizer updates it in place, which bumps `tensor._version`) and
    folds them in at the next new version.  Repeated forward passes on the
    same weights (gradient accumulation, recomputation under gradient
    checkpointing) therefore choose the same levels and count once.
  * bake(model) sets every latent weight to the centre of its chosen level,
    w = k * bf16(s), before the checkpoint is saved.  Plain rounding then gives
    the same levels, so tft/export_weights_poc.py (whose checks compare with
    round to nearest) exports the ECQ levels unchanged.

The choice between neighbours alone cannot make the steps coarser.  The rate
of a weight falls by 1 bit when its row scale doubles (entropy of a fine
quantizer: h(w) - log2 s), so the second switch adds that rate to the training
objective with the exchange rate of the score: the loss in nats per token plus
mu * (bits of all int4 levels).  The archive grows by about 41 KB per 0.001
nat/token (training/tf/tree_overlay/zctx/eval_zctx.py) and a bit of the
container costs 2/8 B of S (it ships twice), so mu = 0.25 / 41e6 = 6.1e-9
prices the two alike.  The term enters the gradient of every row scale as
-mu * n / (s ln 2), scaled to the sum-of-tokens loss of pysrc/train.py, and
with his LSQ normalisation like the rest of that gradient.  n counts the
weights of the row off level 0: a weight on level 0 stays there when s grows,
and a row with all levels 0 pays the same whatever its scale.  (With d_in in
place of n, the term pushed the scales of such rows up by one AdamW step,
about lr, per optimizer step: the zero rows of a container checkpoint start
at s = 2^-24, see tf/4_export/container_to_tch.py, and grew 40-fold in one
step of the CPU smoke test.)

Rows whose latent weights are all zero when install() runs (the zeroed rows
and dead neurons of the shipped container) are frozen by default: hooks zero
the gradients of their weights and of their scale, so they stay what the
container has.  Without the hooks the straight-through gradient moves their
weights by about lr per step, which at s = 2^-24 is a jump to level +-7.

Switches (tft/runner.py reads them through install_from_env):
  TFT_ECQ_LAMBDA   lambda; unset or empty: ECQ off
  TFT_ECQ_RATE_MU  mu in nats per token per bit (0: off; 6.1e-9 is the score's
                   own exchange rate)
  TFT_ECQ_DECAY    EMA factor of the histogram per optimizer step (0.98)
  TFT_ECQ_SCOPE    "tensor" (one histogram per tensor, the default) or "global"
                   (one for all; the container's coder adapts within a tensor,
                   so "tensor" is closer to what it pays)
  TFT_ECQ_FREEZE_ZERO_ROWS  "0" lets the all-zero rows train (default "1")
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor
from torch.autograd import Function

from pysrc.quantization import Quantization, Quantize

LEVELS = 15
P_FLOOR = 1e-6


def _scaled(blocks: Tensor, flat_scale: Tensor, quant: Quantization) -> tuple[Tensor, Tensor]:
    """w/s and s exactly as QuantizeFunction.forward computes them"""
    assert quant.epsilon is None
    b = blocks.to(quant.arithmetic_dtype)
    scale = flat_scale.unsqueeze(0).unsqueeze(-1).to(quant.scale_dtype).type_as(b)
    return b / scale, scale


@torch.no_grad()
def choose_levels(scaled: Tensor, quant: Quantization, cost: Tensor | None) -> Tensor:
    """the ECQ level of every element of `scaled` (= w/s), as a float tensor

    cost: (15,) lambda * bits per level, or None for round to nearest."""
    qmin, qmax = quant.qmin, quant.qmax
    nearest = scaled.round().clamp(qmin, qmax)
    if cost is None:
        return nearest
    lo = scaled.floor().clamp(qmin, qmax)
    hi = (lo + 1).clamp(qmin, qmax)
    c = cost.to(scaled.dtype)
    d_lo = (scaled - lo).square() + c[(lo - qmin).long()]
    d_hi = (scaled - hi).square() + c[(hi - qmin).long()]
    even_hi = torch.remainder(hi, 2) == 0
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & even_hi)
    chosen = torch.where(take_hi, hi, lo)
    inside = (scaled >= qmin) & (scaled <= qmax)
    return torch.where(inside, chosen, nearest)


class ECQFunction(Function):
    """QuantizeFunction with given levels; backward in his LSQ form"""

    @staticmethod
    def forward(ctx, blocks, flat_scale, levels, quant, return_ints,  # type: ignore[override]
                rate_coeff=0.0):
        input_dtype = blocks.dtype
        scaled, scale = _scaled(blocks, flat_scale, quant)
        ctx.save_for_backward(scaled, levels, scale)
        ctx.quant = quant
        ctx.rate_coeff = float(rate_coeff)
        if return_ints:
            return levels.to(torch.int64)
        return (scale * levels).to(input_dtype)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        scaled, levels, scale = ctx.saved_tensors
        quant: Quantization = ctx.quant
        grad_dtype = grad_out.dtype
        g = grad_out.to(quant.arithmetic_dtype)
        too_small = scaled < quant.qmin
        too_big = quant.qmax < scaled
        within = ~(too_small | too_big)
        grad_input = g * within.type_as(g)
        grad_scale = (
            g * (too_small.type_as(g) * quant.qmin
                 + too_big.type_as(g) * quant.qmax
                 + within.type_as(g) * (levels - scaled))
        ).sum(0).sum(-1)
        if ctx.rate_coeff:
            # d/ds of rate_coeff * (bits of the row) = -rate_coeff * n / (s ln 2),
            # n = the row's weights off level 0 (see the module docstring)
            s = scale.reshape(-1)
            n_off = (levels != 0).type_as(g).sum(0).sum(-1)
            grad_scale = grad_scale - ctx.rate_coeff * n_off / (s * math.log(2.0))
        grad_scale = grad_scale / math.sqrt(scaled.numel() * quant.qmax)
        return (grad_input.to(grad_dtype), grad_scale.to(grad_dtype),
                None, None, None, None)


class _Shared:
    """one histogram for several quantizers (TFT_ECQ_SCOPE=global)"""

    def __init__(self, hist: Tensor) -> None:
        self.hist = hist
        self.pending = torch.zeros_like(hist)


class ECQQuantize(Quantize):
    """a 15-level weight quantizer with entropy-constrained rounding; never
    constructed, install() swaps the class of existing instances"""

    ecq_lambda: float
    ecq_decay: float
    ecq_rate_coeff: float
    ecq_shared: _Shared | None

    def ecq_cost(self) -> Tensor:
        hist = self.ecq_shared.hist if self.ecq_shared is not None else self.ecq_hist
        p = hist / hist.sum().clamp_min(1e-30)
        return self.ecq_lambda * -torch.log2(p.clamp_min(P_FLOOR))

    @torch.no_grad()
    def _ecq_new_version(self, version: int) -> None:
        """the optimizer moved the weights: fold in the counts of the last
        version, start counting the new one"""
        if self.ecq_shared is not None:
            sh = self.ecq_shared
            if sh.pending.sum() > 0 and self._ecq_folds_shared:
                sh.hist.mul_(self.ecq_decay).add_(
                    sh.pending / sh.pending.sum(), alpha=1.0 - self.ecq_decay)
                sh.pending.zero_()
        elif self.ecq_pending.sum() > 0:
            self.ecq_hist.mul_(self.ecq_decay).add_(
                self.ecq_pending / self.ecq_pending.sum(), alpha=1.0 - self.ecq_decay)
            self.ecq_pending.zero_()
        self.ecq_version = version
        self.ecq_counted = False

    def forward(self, input: Tensor, initialize_scales_steps: int | None,
                return_ints: bool) -> Tensor:
        if initialize_scales_steps is not None:
            return Quantize.forward(self, input, initialize_scales_steps, return_ints)
        blocks = self.reshape_input(input)
        flat_scale = self.scale.view(-1)
        if self.training and input._version != self.ecq_version:
            self._ecq_new_version(input._version)
        with torch.no_grad():
            scaled, _ = _scaled(blocks, flat_scale, self.quant)
            levels = choose_levels(scaled, self.quant, self.ecq_cost())
            if self.training and not self.ecq_counted:
                counts = torch.bincount(
                    (levels - self.quant.qmin).long().flatten(), minlength=LEVELS
                ).to(self.ecq_hist.dtype)
                if self.ecq_shared is not None:
                    self.ecq_shared.pending.add_(counts)
                else:
                    self.ecq_pending.add_(counts)
                self.ecq_counted = True
        out = ECQFunction.apply(blocks, flat_scale, levels, self.quant, return_ints,
                                self.ecq_rate_coeff if self.training else 0.0)
        return out.view(input.shape)


def weight_quantizers(model: torch.nn.Module):
    """(name, owner module, its 15-level weight quantizer), in module order"""
    for name, module in model.named_modules():
        q = getattr(module, "quantize_weight", None)
        if isinstance(q, Quantize) and q.quant.buckets == LEVELS:
            yield name, module, q


@torch.no_grad()
def _nearest_hist(owner: torch.nn.Module, q: Quantize) -> Tensor:
    blocks = q.reshape_input(owner.weight)
    scaled, _ = _scaled(blocks, q.scale.view(-1), q.quant)
    levels = choose_levels(scaled, q.quant, None)
    return torch.bincount((levels - q.quant.qmin).long().flatten(),
                          minlength=LEVELS).double()


@torch.no_grad()
def _freeze_zero_rows(owner: torch.nn.Module, q: Quantize) -> int:
    """gradient hooks that keep the rows with all latent weights zero (weights
    and scale) as they are; returns the number of such rows"""
    blocks = q.reshape_input(owner.weight)             # a view: (1, rows, d_in)
    zero = (blocks == 0).all(dim=-1)
    n = int(zero.sum())
    if n == 0:
        return 0
    keep = ~zero
    keep_w = keep.unsqueeze(-1).expand_as(blocks).reshape(owner.weight.shape)
    keep_w = keep_w.to(owner.weight.dtype)
    keep_s = keep.reshape(q.scale.shape).to(q.scale.dtype)
    owner.weight.register_hook(lambda g, m=keep_w: g * m.to(g.dtype))
    q.scale.register_hook(lambda g, m=keep_s: g * m.to(g.dtype))
    return n


def install(model: torch.nn.Module, lam: float, decay: float = 0.98,
            scope: str = "tensor", verbose: bool = True,
            rate_coeff: float = 0.0, freeze_zero_rows: bool = True) -> int:
    """ECQ on every 15-level weight quantizer of `model`; returns their number.
    The histograms start from round to nearest on the current weights.
    rate_coeff multiplies d(bits)/d(scale) in every backward pass (0: off).
    freeze_zero_rows: rows whose latent weights are all zero now keep their
    weights and scale (module docstring)."""
    assert lam >= 0 and 0 < decay < 1 and scope in ("tensor", "global")
    assert rate_coeff >= 0
    found = list(weight_quantizers(model))
    frozen = sum(_freeze_zero_rows(o, q) for _n, o, q in found) if freeze_zero_rows else 0
    shared = None
    if scope == "global":
        total = sum((_nearest_hist(o, q) for _n, o, q in found),
                    torch.zeros(LEVELS, dtype=torch.float64))
        dev = found[0][2].scale.device if found else "cpu"
        shared = _Shared((total / total.sum()).to(dev))
    for i, (_name, owner, q) in enumerate(found):
        h = _nearest_hist(owner, q)
        q.__class__ = ECQQuantize
        q.ecq_lambda = float(lam)
        q.ecq_decay = float(decay)
        q.ecq_rate_coeff = float(rate_coeff)
        q.ecq_shared = shared
        q._ecq_folds_shared = i == 0      # one quantizer folds the shared counts
        q.register_buffer("ecq_hist", (h / h.sum()).to(q.scale.device),
                          persistent=False)
        q.register_buffer("ecq_pending", torch.zeros_like(q.ecq_hist),
                          persistent=False)
        q.ecq_version = -1
        q.ecq_counted = False
    if verbose:
        bits, n = rate(model)
        print("ecq: lambda %g, decay %g, scope %s, rate coefficient %g, %d "
              "quantizers, %d weights, start %.4f bits/weight (%.0f B at the "
              "order-0 bound), %d all-zero rows frozen"
              % (lam, decay, scope, rate_coeff, len(found), n, bits / max(n, 1),
                 bits / 8, frozen), flush=True)
    return len(found)


def install_from_env(model: torch.nn.Module, loss_tokens: int = 1,
                     accumulation: int = 1, verbose: bool = True) -> bool:
    """install() from the TFT_ECQ_* switches.  loss_tokens and accumulation:
    the training loss of pysrc/train.py is the SUM over the tokens of a micro
    batch divided by the accumulation steps, so mu (per token) becomes
    mu * loss_tokens / accumulation per backward pass."""
    lam = os.environ.get("TFT_ECQ_LAMBDA", "")
    if not lam:
        return False
    mu = float(os.environ.get("TFT_ECQ_RATE_MU", "0"))
    install(model, float(lam), float(os.environ.get("TFT_ECQ_DECAY", "0.98")),
            os.environ.get("TFT_ECQ_SCOPE", "tensor"), verbose,
            rate_coeff=mu * float(loss_tokens) / float(accumulation),
            freeze_zero_rows=os.environ.get("TFT_ECQ_FREEZE_ZERO_ROWS", "1") != "0")
    return True


@torch.no_grad()
def levels_of(owner: torch.nn.Module, q: Quantize) -> Tensor:
    """the levels the quantizer chooses for the owner's current weight"""
    blocks = q.reshape_input(owner.weight)
    scaled, _ = _scaled(blocks, q.scale.view(-1), q.quant)
    cost = q.ecq_cost() if isinstance(q, ECQQuantize) else None
    return choose_levels(scaled, q.quant, cost)


@torch.no_grad()
def rate(model: torch.nn.Module) -> tuple[float, int]:
    """order-0 entropy of the chosen levels, summed over the tensors
    (bits, weights): the rate the container's adaptive coder approaches"""
    bits = 0.0
    n = 0
    for _name, owner, q in weight_quantizers(model):
        lv = levels_of(owner, q)
        h = torch.bincount((lv - q.quant.qmin).long().flatten(), minlength=LEVELS).double()
        p = h[h > 0] / h.sum()
        bits += float(-(p * torch.log2(p)).sum() * h.sum())
        n += lv.numel()
    return bits, n


@torch.no_grad()
def bake(model: torch.nn.Module, verbose: bool = True) -> int:
    """every latent weight to the centre of its chosen level (see the module
    docstring); returns the number of weights that changed level against
    round to nearest"""
    moved = 0
    for name, owner, q in weight_quantizers(model):
        lv = levels_of(owner, q)
        blocks = q.reshape_input(owner.weight)
        _scaled_now, scale = _scaled(blocks, q.scale.view(-1), q.quant)
        nearest = choose_levels(_scaled_now, q.quant, None)
        moved += int((lv != nearest).sum())
        centre = (scale * lv).to(owner.weight.dtype).view(owner.weight.shape)
        owner.weight.copy_(centre)
        again, _ = _scaled(q.reshape_input(owner.weight), q.scale.view(-1), q.quant)
        assert torch.equal(choose_levels(again, q.quant, None), lv), (
            "bake: round to nearest does not give the chosen levels back in " + name)
    if verbose:
        bits, n = rate(model)
        print("ecq: baked, %d of %d weights off the nearest level, %.4f bits/weight "
              "(%.0f B at the order-0 bound)" % (moved, n, bits / max(n, 1), bits / 8),
              flush=True)
    return moved
