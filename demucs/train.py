#!/usr/bin/env python3
# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training script entry point"""

import logging
import os
from pathlib import Path
import sys

from dora import hydra_main
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch

from demucs import distrib
from demucs.wav import get_wav_datasets
from demucs.demucs import Demucs
from demucs.hdemucs import HDemucs
from demucs.htdemucs import HTDemucs
from demucs.solver import Solver
from demucs.utils import random_subset
from demucs.speech import get_librimix_wav_datasets

logger = logging.getLogger(__name__)


def get_model(sources: list[str], channels: int, samplerate: int, segment_length: int, model_name: str, args):
    extra = {
        'sources': sources,
        'audio_channels': channels,
        'samplerate': samplerate,
        'segment': segment_length,
    }
    klass = {
        'demucs': Demucs,
        'hdemucs': HDemucs,
        'htdemucs': HTDemucs,
    }[model_name]
    kw = OmegaConf.to_container(getattr(args, model_name), resolve=True)
    model = klass(**extra, **kw)
    return model


def get_optimizer(model, args):
    seen_params = set()
    other_params = []
    groups = []
    for n, module in model.named_modules():
        if hasattr(module, "make_optim_group"):
            group = module.make_optim_group()
            params = set(group["params"])
            assert params.isdisjoint(seen_params)
            seen_params |= set(params)
            groups.append(group)
    for param in model.parameters():
        if param not in seen_params:
            other_params.append(param)
    groups.insert(0, {"params": other_params})
    parameters = groups
    if args.optim.optim == "adam":
        return torch.optim.Adam(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    elif args.optim.optim == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer %s", args.optim.optimizer)


def get_datasets(args_dset, train_data_root: Path, sources: list[str], samplerate: int, segment_length: int, audio_channels: int):
    train_set, valid_set = get_librimix_wav_datasets(train_data_root, sources=sources, samplerate=samplerate, segment=segment_length, audio_channels=audio_channels)
    if args_dset.valid_samples is not None:
        valid_set = random_subset(valid_set, args_dset.valid_samples)
    return train_set, valid_set


def get_solver(args, model_only=False) -> Solver:
    distrib.init()

    torch.manual_seed(args.seed)
    segment_length = args.model_segment or 4 * args.dset.segment
    model = get_model(list(args.dset.sources), args.dset.channels, args.dset.samplerate, segment_length, args.model, args)
    if args.misc.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.dset.samplerate * 1000)
        sys.exit(0)

    # torch also initialize cuda seed if available
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = get_optimizer(model, args)

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    if model_only:
        return Solver(None, model, optimizer, args)

    train_set, valid_set = get_datasets(args.dset, Path(args.dset.root), args.dset.sources, args.dset.samplerate, args.dset.segment, args.dset.channels)
    # if args.augment.repitch.proba:
    #     vocals = []
    #     if 'vocals' in args.dset.sources:
    #         vocals.append(args.dset.sources.index('vocals'))
    #     else:
    #         logger.warning('No vocal source found')
    #     if args.augment.repitch.proba:
    #         train_set = RepitchedWrapper(train_set, vocals=vocals, **args.augment.repitch)

    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    train_loader = distrib.loader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.misc.num_workers, drop_last=True,
        persistent_workers=True, prefetch_factor=32)

    if args.dset.full_cv:
        valid_loader = distrib.loader(
            valid_set, batch_size=1, shuffle=False,
            num_workers=args.misc.num_workers,
            persistent_workers=True, prefetch_factor=32)
    else:
        valid_loader = distrib.loader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.misc.num_workers, drop_last=True,
            persistent_workers=True, prefetch_factor=32)

    loaders = {"train": train_loader, "valid": valid_loader}

    # Construct Solver
    return Solver(loaders, model, optimizer, args)


def get_solver_from_sig(sig, model_only=False):
    inst = GlobalHydra.instance()
    hyd = None
    if inst.is_initialized():
        hyd = inst.hydra
        inst.clear()
    xp = main.get_xp_from_sig(sig)
    if hyd is not None:
        inst.clear()
        inst.initialize(hyd)

    with xp.enter(stack=True):
        return get_solver(xp.cfg, model_only)


@hydra_main(config_path="../conf", config_name="config")
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    # for attr in ["metadata"]:
    #     val = getattr(args.dset, attr)
    #     if val is not None:
    #         setattr(args.dset, attr, hydra.utils.to_absolute_path(val))

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.misc.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    from dora import get_xp
    logger.debug(get_xp().cfg)

    args.dset.segment = 15
    args.batch_size = 32
    args.dset.samplerate = 8000
    args.dset.root = "/work3/projects/02456/project04/librimix/Libri2Mix/wav16k/max/train-360/"
    solver = get_solver(args)
    solver.train()


if '_DORA_TEST_PATH' in os.environ:
    main.dora.dir = Path(os.environ['_DORA_TEST_PATH'])


if __name__ == "__main__":
    main()
