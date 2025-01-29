r"""Attention layers."""

__all__ = [
    "MultiheadSelfAttention",
]

import torch
import torch.nn as nn
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import xformers.components.attention.core as xfa
    import xformers.sparse as xfs

from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union


class MultiheadSelfAttention(nn.Module):
    r"""Creates a multi-head self-attention layer.

    Arguments:
        channels: The number of channels :math:`H \times C`.
        attention_heads: The number of attention heads :math:`H`.
        qk_norm: Whether to use query-key RMS-normalization or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        attention_heads: int = 1,
        qk_norm: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ):
        super().__init__()

        assert channels % attention_heads == 0

        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=False)
        self.y_proj = nn.Linear(channels, channels)

        if qk_norm:
            self.qk_norm = nn.RMSNorm(
                channels // attention_heads,
                elementwise_affine=False,
                eps=1e-5,
            )
        else:
            self.qk_norm = nn.Identity()

        self.heads = attention_heads
        self.dropout = nn.Dropout(0.0 if dropout is None else dropout)
        self.checkpointing = checkpointing

    def _forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Union[Tensor, xfs.SparseCSRTensor]] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, H \times C)`.
            theta: Optional rotary positional embedding :math:`\theta`,
                with shape :math:`(*, L, H \times C / 2)`.
            mask: Optional attention mask, with shape :math:`(L, L)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, H \times C)`.
        """

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... L (n H C) -> n ... H L C", n=3, H=self.heads)
        q, k = self.qk_norm(q), self.qk_norm(k)

        if theta is not None:
            theta = rearrange(theta, "... L (H C) -> ... H L C", H=self.heads)
            q, k = apply_rope(q, k, theta)

        if isinstance(mask, xfs.SparseCSRTensor):
            y = xfa.scaled_dot_product_attention(
                q=rearrange(q, "... L C -> (...) L C"),
                k=rearrange(k, "... L C -> (...) L C"),
                v=rearrange(v, "... L C -> (...) L C"),
                att_mask=xfa.SparseCS._wrap(mask),
                dropout=self.dropout if self.training else None,
            )
            y = y.reshape(q.shape[:-2] + y.shape[-2:])
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0,
            )

        y = rearrange(y, "... H L C -> ... L (H C)")
        y = self.y_proj(y)

        return y

    def forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Union[Tensor, xfs.SparseCSRTensor]] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, theta, mask, use_reentrant=False)
        else:
            return self._forward(x, theta, mask)


def apply_rope(q: Tensor, k: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    References:
        | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
        | https://arxiv.org/abs/2104.09864

        | Rotary Position Embedding for Vision Transformer (Heo et al., 2024)
        | https://arxiv.org/abs/2403.13298

    Arguments:
        q: The query tokens :math:`q`, with shape :math:`(*, C)`.
        k: The key tokens :math:`k`, with shape :math:`(*, C)`.
        theta: Rotary angles, with shape :math:`(*, C / 2)`.

    Returns:
        The rotated query and key tokens, with shape :math:`(*, C)`.
    """

    rotation = torch.polar(torch.ones_like(theta), theta)

    q = torch.view_as_complex(torch.unflatten(q, -1, (-1, 2)))
    k = torch.view_as_complex(torch.unflatten(k, -1, (-1, 2)))

    q = torch.flatten(torch.view_as_real(rotation * q), -2)
    k = torch.flatten(torch.view_as_real(rotation * k), -2)

    return q, k


# fmt: off
def monkey_coo_to_csr(m, n, row_indices, column_indices):
    row_offsets = row_indices.bincount(minlength=m).cumsum(0, dtype=row_indices.dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))

    return row_offsets, column_indices

def monkey_round_nnz(mask, divisible_by=4):
    nnz = torch.count_nonzero(mask)
    cunz = torch.cumsum(~mask.flatten(), dim=0)
    flip = cunz <= (-nnz) % divisible_by

    return torch.logical_or(mask, flip.reshape_as(mask))

def monkey_masked_matmul(cls, a, b, mask):
    assert mask.shape[1] == a.shape[1]
    assert mask.shape[2] == b.shape[2]

    values = mask._SparseCSRTensor__values
    row_indices = mask._SparseCSRTensor__row_indices
    row_offsets = mask._SparseCSRTensor__row_offsets
    column_indices = mask._SparseCSRTensor__column_indices
    tansp_info = mask._SparseCSRTensor__transp_info

    out = xfs._csr_ops._sddmm.apply(
        a.contiguous(),
        b.transpose(-2, -1).contiguous(),
        row_indices,
        row_offsets,
        column_indices,
        tansp_info,
    )
    out = torch.where(values, out, float("-inf"))

    return cls._wrap(
        mask.shape,
        out,
        row_indices,
        row_offsets,
        column_indices,
        tansp_info,
    )

xfs.utils._coo_to_csr = monkey_coo_to_csr
xfs.utils._round_nnz = monkey_round_nnz
xfs.SparseCSRTensor._masked_matmul = classmethod(monkey_masked_matmul)
# fmt: on
