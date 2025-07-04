r"""Optimization and training helpers."""

__all__ = [
    "ExponentialMovingAverage",
    "get_optimizer",
    "safe_gd_step",
]

import heavyball
import math
import torch
import torch.nn as nn

from functools import partial
from heavyball import ForeachCachedDelayedPSGDKron, ForeachSFAdamW, ForeachSOAP
from torch import Tensor
from typing import Iterable, Optional, Sequence, Tuple

from .soap import SOAP

heavyball.utils.compile_mode = "default"


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    r"""Creates an exponential moving average (EMA) module.

    Arguments:
        module: The averaged module.
        decay: The exponential decay in [0, 1]. If :py:`None`, averaging is skipped.
    """

    def __init__(
        self,
        module: nn.Module,
        decay: Optional[float] = None,
    ):
        if decay is None:
            module = None
            multi_avg_fn = None
        else:
            multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay)

        super().__init__(
            model=module,
            multi_avg_fn=multi_avg_fn,
        )

    def update_parameters(self, module: nn.Module):
        if self.multi_avg_fn is None:
            self.module = module
        else:
            super().update_parameters(module)


def precond_prob_schedule(n, max_prob=1.0, min_prob=0.01, decay=0.999, flat_start=0):
    return max(min_prob, max_prob * decay ** max(n - flat_start, 0))


def get_optimizer(
    params: Iterable[nn.Parameter],
    optimizer: str = "adamw",
    betas: Sequence[float] = (0.9, 0.99, 0.99),
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    scheduler: Optional[str] = None,
    epochs: Optional[int] = None,
    warmup: Optional[int] = None,
    # SOAP & PSGD
    precondition_frequency: int = 16,
    precondition_frequency_decay: float = 0.999,
    precondition_warmup: int = 0,
    precondition_size: int = 4096,
    merge_dims: bool = False,
    # Ignored
    name: str = None,
    grad_clip: float = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    r"""Instantiates an optimizer and sheduler.

    Arguments:
        params: The optimized parameters.
        optimizer: The optimizer name.
        learning_rate: The learning rate.
        weight_decay: The weight decay.
        scheduler: The scheduler name.
        epochs: The total number of epochs.
        warmup: The number of warmup epochs.

    Returns:
        An optimizer/scheduler pair.
    """

    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=betas[:2],
            weight_decay=weight_decay,
        )
    elif optimizer == "adamw-sf":
        optimizer = ForeachSFAdamW(
            params,
            lr=learning_rate,
            betas=betas[:2],
            weight_decay=weight_decay,
        )
    elif optimizer == "soap":
        optimizer = SOAP(
            params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            precondition_warmup=precondition_warmup,
            max_precond_size=precondition_size,
            merge_dims=merge_dims,
        )
    elif optimizer == "soap-foreach":
        optimizer = ForeachSOAP(
            params,
            lr=learning_rate,
            betas=betas[:2],
            shampoo_beta=betas[2],
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=precondition_size,
            merge_dims=merge_dims,
        )
    elif optimizer == "psgd":
        optimizer = ForeachCachedDelayedPSGDKron(
            params,
            lr=learning_rate,
            beta=betas[0],
            weight_decay=weight_decay,
            preconditioner_update_probability=partial(
                precond_prob_schedule,
                min_prob=1 / precondition_frequency,
                decay=precondition_frequency_decay,
            ),
            max_size_triangular=precondition_size,
            merge_dims=merge_dims,
        )
    else:
        raise NotImplementedError()

    if scheduler is None:
        lr_lambda = lambda t: 1
    elif scheduler == "linear":
        lr_lambda = lambda t: max(0, 1 - (t / epochs))
    elif scheduler == "cosine":
        lr_lambda = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif scheduler == "exponential":
        lr_lambda = lambda t: math.exp(math.log(1e-3) * t / epochs)
    else:
        raise NotImplementedError()

    if warmup is None:
        cold_lr_lambda = lr_lambda
    else:
        cold_lr_lambda = lambda t: min(1, (t + 1) / (warmup + 1)) * lr_lambda(t)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cold_lr_lambda)

    return optimizer, scheduler


def safe_gd_step(
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
) -> Tensor:
    r"""Applies a gradient descent (GD) optimization step.

    To prevent invalid parameters, steps are skipped if not-a-number (NaN) or infinite
    values are found in the gradient. This feature requires CPU-GPU synchronization,
    which could be a bottleneck for some applications.

    Arguments:
        optimizer: An optimizer.
        grad_clip: The maximum gradient norm. If :py:`None`, gradients are not clipped.

    Returns:
        The unclipped gradient norm.
    """

    params = [p for group in optimizer.param_groups for p in group["params"]]

    if grad_clip is None:
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad) for p in params if torch.is_tensor(p.grad)])
        )
    else:
        norm = nn.utils.clip_grad_norm_(params, grad_clip)

    if norm.isfinite():
        optimizer.step()

    optimizer.zero_grad()

    return norm.detach()
