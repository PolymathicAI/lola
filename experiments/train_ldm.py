#!/usr/bin/env python

import argparse
import dawgz
import wandb

from functools import partial
from omegaconf import DictConfig

from lpdm.hydra import compose


def train(runid: str, cfg: DictConfig):
    import os
    import torch
    import torch.distributed as dist
    import wandb

    from einops import rearrange
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import trange

    from lpdm.data import MiniWellDataset, find_hdf5, get_dataloader, get_well_inputs
    from lpdm.diffusion import DenoiserLoss, get_denoiser
    from lpdm.emulation import random_context_mask
    from lpdm.optim import ExponentialMovingAverage, get_optimizer, safe_gd_step
    from lpdm.utils import randseed

    # DDP
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_id = os.environ.get("LOCAL_RANK", rank)
    device_id = int(device_id)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Config
    assert cfg.train.epoch_size % cfg.train.batch_size == 0
    assert cfg.train.batch_size % (cfg.train.accumulation * world_size) == 0

    runname = f"{runid}_{cfg.dataset.name}_{cfg.denoiser.name}"

    runpath = Path(f"{cfg.server.storage}/runs/ldm/{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        os.symlink(os.path.realpath(cfg.ae_run, strict=True), runpath / "autoencoder")

    dist.barrier(device_ids=[device_id])

    with open_dict(cfg):
        cfg.name = runname
        cfg.path = str(runpath)
        cfg.seed = randseed(runid)

    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # Stem
    if cfg.fork.run is None:
        counter = {
            "epoch": 0,
            "update_samples": 0,
            "update_steps": 0,
        }
    else:
        stem = wandb.Api().run(path=cfg.fork.run)
        stem_name = Path(stem.config["path"]).name
        stem_path = Path(f"{cfg.server.storage}/runs/ldm/{stem_name}")
        stem_path = stem_path.expanduser().resolve()
        stem_state = torch.load(
            stem_path / f"{cfg.fork.target}.pth", weights_only=True, map_location=device
        )

        counter = {
            "epoch": stem.summary["_step"] + 1,
            "update_samples": stem.summary["train/samples"],
            "update_steps": stem.summary["train/update_steps"],
        }

    # Data
    files = {
        split: [
            file
            for physic in cfg.dataset.physics
            for file in find_hdf5(
                path=runpath / "autoencoder/cache" / physic / split,
                include_filters=cfg.dataset.include_filters,
            )
        ]
        for split in ("train", "valid")
    }

    dataset = {
        split: MiniWellDataset.from_files(
            files=files[split],
            steps=cfg.trajectory.length,
            stride=cfg.trajectory.stride,
        )
        for split in ("train", "valid")
    }

    train_loader, valid_loader = [
        get_dataloader(
            dataset=dataset[split],
            batch_size=cfg.train.batch_size // cfg.train.accumulation // world_size,
            shuffle=True if split == "train" else False,
            infinite=True,
            num_workers=cfg.compute.cpus_per_gpu,
            rank=rank,
            world_size=world_size,
            seed=cfg.seed,
        )
        for split in ("train", "valid")
    ]

    z, label = get_well_inputs(next(valid_loader))
    z = rearrange(z, "B L H W C -> B C L H W")

    # Model, optimizer & scheduler
    denoiser = get_denoiser(
        shape=z.shape[1:],
        label_features=label.shape[1],
        masked=True,
        **cfg.denoiser,
    ).to(device)

    denoiser_loss = DenoiserLoss(**cfg.denoiser.loss).to(device)

    if cfg.fork.run is not None:
        denoiser.load_state_dict(stem_state, strict=cfg.fork.strict)
        del stem_state

    denoiser = DistributedDataParallel(
        module=denoiser,
        device_ids=[device_id],
    )

    optimizer, scheduler = get_optimizer(
        params=denoiser.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
    )

    average = ExponentialMovingAverage(
        module=denoiser.module,
        decay=cfg.train.ema_decay,
    )

    # W&B
    if rank == 0:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project="mpp-ldm",
            id=runid,
            name=runname,
            config=OmegaConf.to_container(cfg),
        )

    # Training loop
    if rank == 0:
        epochs = trange(cfg.train.epochs, ncols=88, ascii=True)
    else:
        epochs = range(cfg.train.epochs)

    best_valid_loss = float("inf")

    for _ in epochs:
        ## Train
        denoiser.train()

        losses, grads = [], []

        for i in range(cfg.train.epoch_size // cfg.train.batch_size):
            z, label = get_well_inputs(next(train_loader), device=device)
            z = rearrange(z, "B L H W C -> B C L H W")

            mask = random_context_mask(z, **cfg.trajectory.context)

            if (i + 1) % cfg.train.accumulation == 0:
                loss = denoiser_loss(denoiser, z, mask=mask, label=label)
                loss_acc = loss / cfg.train.accumulation
                loss_acc.backward()

                grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)
                grads.append(grad_norm)

                average.update_parameters(denoiser.module)

                counter["update_samples"] += cfg.train.batch_size
                counter["update_steps"] += 1
            else:
                with denoiser.no_sync():
                    loss = denoiser_loss(denoiser, z, mask=mask, label=label)
                    loss_acc = loss / cfg.train.accumulation
                    loss_acc.backward()

            losses.append(loss.detach())

        losses = torch.stack(losses)
        grads = torch.stack(grads)

        if rank == 0:
            losses_list = [torch.empty_like(losses) for _ in range(world_size)]
            grads_list = [torch.empty_like(grads) for _ in range(world_size)]
        else:
            losses_list = None
            grads_list = None

        dist.gather(losses, losses_list, dst=0)
        dist.gather(grads, grads_list, dst=0)

        if rank == 0:
            losses = torch.cat(losses_list).cpu()
            grads = torch.cat(grads_list).cpu()

            logs = {}
            logs["train/loss/mean"] = losses.mean().item()
            logs["train/loss/std"] = losses.std().item()
            logs["train/grad_norm/mean"] = grads.mean().item()
            logs["train/grad_norm/std"] = grads.std().item()
            logs["train/learning_rate"] = optimizer.param_groups[0]["lr"]
            logs["train/update_steps"] = counter["update_steps"]
            logs["train/samples"] = counter["update_samples"]

        del losses, losses_list, grads, grads_list

        ## Eval
        denoiser.eval()

        losses = []

        with torch.no_grad():
            for _ in range(cfg.train.epoch_size // cfg.train.batch_size):
                z, label = get_well_inputs(next(valid_loader), device=device)
                z = rearrange(z, "B L H W C -> B C L H W")

                mask = random_context_mask(z, **cfg.trajectory.context)

                loss = denoiser_loss(denoiser, z, mask=mask, label=label)
                losses.append(loss)

        losses = torch.stack(losses)

        if rank == 0:
            losses_list = [torch.empty_like(losses) for _ in range(world_size)]
        else:
            losses_list = None

        dist.gather(losses, losses_list, dst=0)

        if rank == 0:
            losses = torch.cat(losses_list).cpu()

            logs["valid/loss/mean"] = losses.mean().item()
            logs["valid/loss/std"] = losses.std().item()

            epochs.set_postfix(
                lt=logs["train/loss/mean"],
                lv=logs["valid/loss/mean"],
            )

            run.log(logs, step=counter["epoch"])

            counter["epoch"] += 1

        del losses, losses_list

        ## LR scheduler
        scheduler.step()

        ## Checkpoint
        if rank == 0:
            state = denoiser.module.state_dict()
            state_ema = average.module.state_dict()

            torch.save(state, runpath / "state.pth")
            torch.save(state_ema, runpath / "state_ema.pth")

            if logs["valid/loss/mean"] < best_valid_loss:
                best_valid_loss = logs["valid/loss/mean"]

                torch.save(state, runpath / "state_best.pth")
                torch.save(state_ema, runpath / "state_best_ema.pth")

            del state
            del state_ema

        dist.barrier(device_ids=[device_id])

    # W&B
    if rank == 0:
        run.finish()

    # DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/train_ldm.yaml",
        overrides=args.overrides,
    )

    # Job
    runid = wandb.util.generate_id()

    dawgz.schedule(
        dawgz.job(
            f=partial(train, runid, cfg),
            name=f"ldm {runid}",
            cpus=cfg.compute.cpus_per_gpu * cfg.compute.gpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
            exclude=cfg.server.exclude,
        ),
        name=f"training ldm {runid}",
        backend="slurm",
        interpreter=f"torchrun --nnodes 1 --nproc-per-node {cfg.compute.gpus} --standalone",
        env=[
            "export OMP_NUM_THREADS=" + f"{cfg.compute.cpus_per_gpu}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
