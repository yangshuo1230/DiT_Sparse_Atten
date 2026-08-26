"""Dense PyTorch SDPA backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DenseBackend:
    name = "dense"

    def __call__(
        self,
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
        **_,
    ):
        del q_lens, k_lens, deterministic, version
        if window_size != (-1, -1):
            raise NotImplementedError("The dense SDPA backend only supports global attention")
        output_dtype = q.dtype
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        if q_scale is not None:
            q = q * q_scale
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        return output.transpose(1, 2).contiguous().to(output_dtype)


def install(backend=None):
    """Install a backend into Wan's module-level attention call sites."""
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module

    backend = backend or DenseBackend()
    attention_module.flash_attention = backend
    model_module.flash_attention = backend
    return backend
