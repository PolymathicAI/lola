#!/usr/bin/env python

import argparse
import dawgz

from functools import partial
from omegaconf import DictConfig

from lpdm.hydra import compose


def get_stats(cfg: DictConfig):
    import itertools
    import torch

    from einops import rearrange
    from tabulate import tabulate
    from torch.utils.data import DataLoader

    from lpdm.data import field_preprocess, get_well_multi_dataset
    from lpdm.utils import process_cpu_count

    # Data
    trainset = get_well_multi_dataset(
        path=cfg.server.datasets,
        physics=cfg.dataset.physics,
        split="train",
        steps=1,
        include_filters=cfg.dataset.include_filters,
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=None,
        shuffle=True,
        num_workers=process_cpu_count(),
    )

    preprocess = partial(
        field_preprocess,
        transform=cfg.dataset.transform,
    )

    # Fetch
    mins = []
    maxs = []
    first_moments = []
    second_moments = []

    for batch in itertools.islice(train_loader, cfg.samples):
        x = batch["input_fields"]
        x = preprocess(x)
        x = rearrange(x, "... C -> (...) C")

        mins.append(x.min(dim=0).values)
        maxs.append(x.max(dim=0).values)
        first_moments.append(x.mean(dim=0))
        second_moments.append(x.square().mean(dim=0))

    mins = torch.stack(mins).min(dim=0).values
    maxs = torch.stack(maxs).max(dim=0).values
    first_moments = torch.stack(first_moments).mean(dim=0)
    second_moments = torch.stack(second_moments).mean(dim=0)

    stats = {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "mean": first_moments.tolist(),
        "std": torch.sqrt(second_moments - first_moments**2).tolist(),
    }

    rows = [(k, *v) for k, v in stats.items()]

    print(tabulate(rows, headers=("Stat", *cfg.dataset.fields)), flush=True)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/get_stats.yaml",
        overrides=args.overrides,
    )

    # Job
    dawgz.schedule(
        dawgz.job(
            f=partial(get_stats, cfg),
            name="stats",
            cpus=cfg.compute.cpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
            exclude=cfg.server.exclude,
        ),
        name=f"stats {cfg.dataset.name}",
        backend="slurm",
    )
