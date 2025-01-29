r"""U-Net building blocks.

Credits:
    https://github.com/probabilists/azula
"""

__all__ = [
    "UNetBlock",
    "UNet",
]

import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Sequence, Union

from .layers import ConvNd, LayerNorm, SelfAttentionNd


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class UNetBlock(nn.Module):
    r"""Creates a modulated U-Net block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        norm: The kind of normalization.
        groups: The number of groups in :class:`torch.nn.GroupNorm` layers.
        attention_heads: The number of attention heads.
        spatial: The number of spatial dimensions :math:`N`.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        norm: str = "group",
        groups: int = 16,
        attention_heads: Optional[int] = None,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Attention
        if attention_heads is None:
            self.attn = nn.Identity()
        else:
            self.attn = Residual(
                LayerNorm(dim=-spatial - 1),
                SelfAttentionNd(channels, heads=attention_heads),
            )

        # Ada-Norm Zero
        if norm == "group":
            self.norm = nn.GroupNorm(
                num_groups=min(groups, channels),
                num_channels=channels,
                affine=False,
            )
        elif norm == "layer":
            self.norm = LayerNorm(dim=-spatial - 1)
        else:
            raise NotImplementedError()

        self.ada_zero = nn.Sequential(
            nn.Linear(mod_features, max(mod_features, channels)),
            nn.SiLU(),
            nn.Linear(max(mod_features, channels), 3 * channels),
            Rearrange("... (n C) -> n ... C" + " 1" * spatial, n=3),
        )

        self.ada_zero[-2].weight.data.mul_(1e-2)

        # Block
        self.block = nn.Sequential(
            ConvNd(channels, channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
        )

    def _forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        a, b, c = self.ada_zero(mod)

        y = self.attn(x)
        y = (a + 1) * self.norm(y) + b
        y = self.block(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        return y

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class UNet(nn.Module):
    r"""Creates a modulated U-Net module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        cond_channels: The number of condition channels :math:`C_c`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        norm: The kind of normalization.
        attention_heads: The number of attention heads at each depth.
        spatial: The number of spatial dimensions :math:`N`.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        mod_features: int = 0,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        norm: str = "group",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        spatial: int = 2,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            do, up = nn.ModuleList(), nn.ModuleList()

            for _ in range(num_blocks):
                do.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

                up.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            if i > 0:
                do.insert(
                    0,
                    ConvNd(
                        hid_channels[i - 1],
                        hid_channels[i],
                        stride=stride,
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    ),
                )

                up.append(nn.Upsample(scale_factor=tuple(stride), mode="nearest"))
            else:
                do.insert(
                    0,
                    ConvNd(
                        in_channels + cond_channels + spatial,
                        hid_channels[i],
                        spatial=spatial,
                        **kwargs,
                    ),
                )
                up.append(ConvNd(hid_channels[i], out_channels, spatial=spatial, **kwargs))

            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(self, x: Tensor, mod: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.
            cond: The condition tensor, with :math:`(B, C_c, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1, ..., L_N)`.
        """

        B, _, *shape = x.shape

        p = (torch.linspace(-1, 1, steps=size, device=x.device) for size in shape)
        p = torch.cartesian_prod(*p)
        p = torch.reshape(p, shape=(*shape, -1))
        p = repeat(p, "... C -> B C ...", B=B)

        if cond is None:
            x = torch.cat((x, p), dim=1)
        else:
            x = torch.cat((x, p, cond), dim=1)

        memory = []

        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

            memory.append(x)

        for blocks in self.ascent:
            y = memory.pop()
            if x is not y:
                for i in range(2, x.ndim):
                    if x.shape[i] > y.shape[i]:
                        x = torch.narrow(x, i, 0, y.shape[i])

                x = torch.cat((x, y), dim=1)

            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

        return x
