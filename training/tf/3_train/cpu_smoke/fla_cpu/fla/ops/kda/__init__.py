# CPU stand-in for fla.ops.kda.chunk_kda: the pure-torch per-step reference of
# training/tf/tree_upstream/pysrc/kda_reference.py (slow, exact semantics)
import torch


def chunk_kda(q, k, v, g, beta, A_log, dt_bias, use_qk_l2norm_in_kernel=True,
              use_gate_in_kernel=True, cu_seqlens=None, **kw):
    from pysrc.kda_reference import kda_recurrent_reference
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel
    outs = [kda_recurrent_reference(q[b:b + 1].float(), k[b:b + 1].float(), v[b:b + 1].float(),
                                    g[b:b + 1].float(), beta[b:b + 1].float(), A_log, dt_bias,
                                    cu_seqlens, dtype=torch.float32)
            for b in range(q.shape[0])]
    return torch.cat(outs, 0).to(q.dtype), None
