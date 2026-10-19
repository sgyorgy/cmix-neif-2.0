# CPU stand-in for fla's causal_conv1d (activation "silu", no bias)
import torch


def causal_conv1d(x, weight, activation="silu", cu_seqlens=None, **k):
    from pysrc.kda_reference import causal_conv1d_silu_reference
    assert activation == "silu"
    outs = [causal_conv1d_silu_reference(x[b:b + 1].float(), weight, cu_seqlens, dtype=torch.float32)
            for b in range(x.shape[0])]
    return torch.cat(outs, 0).to(x.dtype), None
