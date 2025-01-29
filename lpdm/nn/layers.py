r"""Common layers and modules."""

__all__ = [
    "ConvNd",
    "LayerNorm",
    "RMSNorm",
    "SelfAttentionNd",
    "Patchify",
    "Unpatchify",
]

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Sequence, Union


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        spatial: The number of spatial dimensions :math:`N`.
        identity_init: Initialize the convolution as a (pseudo-)identity.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()

    conv = Conv(in_channels, out_channels, **kwargs)

    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]

        eye = torch.zeros_like(conv.weight.data)

        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1

        conv.weight.data.add_(eye)

    return conv


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The standardized tensor :math:`y`, with shape :math:`(*)`.
        """

        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)

        return (x - mean) * torch.rsqrt(variance + self.eps)


class RMSNorm(nn.Module):
    r"""Creates a layer that normalizes features along a dimension.

    .. math:: y = \frac{x}{\sqrt{\mathbb{E}[x^2] + \epsilon}}

    References:
       | Root Mean Square Layer Normalization (Zhang et al., 2019)
       | https://arxiv.org/abs/1910.07467

    Arguments:
        dim: The dimension(s) to normalize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The normalized tensor :math:`y`, with shape :math:`(*)`.
        """

        rms = torch.rsqrt(torch.mean(torch.square(x), dim=self.dim, keepdim=True) + self.eps)

        return x * rms


class SelfAttentionNd(nn.MultiheadAttention):
    r"""Creates an N-dimensional self-attention layer.

    Arguments:
        channels: The number of channels :math:`C`.
        heads: The number of attention heads.
        kwargs: Keyword arguments passed to :class:`torch.nn.MultiheadAttention`.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)

        self.checkpointing = checkpointing

    def _forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(B, C, L_1, ..., L_N)`.

        Returns:
            The ouput tensor :math:`y`, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        y = rearrange(x, "B C ...  -> B (...) C")
        y, _ = super().forward(y, y, y, average_attn_weights=False)
        y = rearrange(y, "B L C -> B C L").reshape(x.shape)

        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class Patchify(nn.Module):
    def __init__(self, patch_size: Sequence[int]):
        super().__init__()

        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        if len(self.patch_size) == 1:
            (l,) = self.patch_size
            return rearrange(x, "... C (L l) -> ... (C l) L", l=l)
        elif len(self.patch_size) == 2:
            h, w = self.patch_size
            return rearrange(x, "... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
        elif len(self.patch_size) == 3:
            t, h, w = self.patch_size
            return rearrange(x, "... C (T t) (H h) (W w) -> ... (C t h w) T H W", t=t, h=h, w=w)
        else:
            raise NotImplementedError()


class Unpatchify(nn.Module):
    def __init__(self, patch_size: Sequence[int]):
        super().__init__()

        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        if len(self.patch_size) == 1:
            (l,) = self.patch_size
            return rearrange(x, "... (C l) L -> ... C (L l)", l=l)
        elif len(self.patch_size) == 2:
            h, w = self.patch_size
            return rearrange(x, "... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
        elif len(self.patch_size) == 3:
            l, h, w = self.patch_size
            return rearrange(x, "... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
        else:
            raise NotImplementedError()
