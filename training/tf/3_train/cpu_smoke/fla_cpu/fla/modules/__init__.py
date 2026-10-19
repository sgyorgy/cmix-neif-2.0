# CPU stand-in for fla.modules.FusedRMSNormGated (activation "sigmoid")
import torch
from torch.nn import Module, Parameter


class FusedRMSNormGated(Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **k):
        super().__init__()
        assert activation == "sigmoid"
        self.weight = Parameter(torch.ones(hidden_size))

    def forward(self, x, g, *a, **k):
        from pysrc.kda_reference import fused_rmsnorm_gated_reference
        return fused_rmsnorm_gated_reference(x, g, self.weight).to(x.dtype)
