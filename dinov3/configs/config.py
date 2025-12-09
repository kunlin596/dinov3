# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 Configuration Management.

This module provides utilities for loading, merging, and managing DINOv3 training
configurations using OmegaConf. It handles:

- Default configuration loading from YAML files
- Configuration merging (default + user config + CLI overrides)
- Learning rate scaling rules for distributed training
- Multi-distillation setup with process subgroups
- Job initialization (logging, distributed, random seeds)

Configuration Hierarchy:
-----------------------
::

    ┌─────────────────────────────────────────────────────────────┐
    │                  Configuration Merge Order                  │
    │                                                             │
    │   1. ssl_default_config.yaml    (base defaults)             │
    │              ↓                                              │
    │   2. User config file           (--config-file)             │
    │              ↓                                              │
    │   3. CLI overrides              (train.lr=0.001)            │
    │              ↓                                              │
    │   4. Final merged config                                    │
    └─────────────────────────────────────────────────────────────┘

Learning Rate Scaling:
---------------------
Two scaling rules are supported for adjusting learning rate based on
effective batch size:

- ``linear_wrt_256``: Linear scaling relative to batch size 256
- ``sqrt_wrt_1024``: Square root scaling relative to batch size 1024

Usage Example:
-------------
.. code-block:: python

    from dinov3.configs.config import setup_config, setup_job, DinoV3SetupArgs

    # Setup distributed environment and logging
    setup_job(output_dir="./outputs", seed=42)

    # Load and merge configurations
    args = DinoV3SetupArgs(
        config_file="configs/train/vitl_im1k.yaml",
        output_dir="./outputs",
        opts=["train.batch_size_per_gpu=64", "optim.lr=0.001"],
    )
    cfg = setup_config(args)

    # Use cfg for training...
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from omegaconf import DictConfig, OmegaConf

import dinov3.distributed as distributed
from dinov3.logging import cleanup_logging, setup_logging
from dinov3.utils import fix_random_seeds, get_conda_env, get_sha

logger = logging.getLogger("dinov3")


@dataclass
class DinoV3SetupArgs:
    """
    Arguments for DINOv3 training setup.

    This dataclass encapsulates all command-line arguments needed to initialize
    a DINOv3 training run. It handles configuration file paths, output directories,
    and CLI overrides.

    Attributes:
    ----------
    config_file : str
        Path to the YAML configuration file (e.g., ``configs/train/vitl_im1k.yaml``).

    pretrained_weights : str | None
        Path to pretrained weights for initialization. If None, training starts
        from scratch with random initialization.

    shard_unsharded_model : bool
        If True, shard an unsharded model checkpoint across FSDP ranks during loading.
        Useful when loading a single-GPU checkpoint for distributed training.

    output_dir : str
        Directory for saving checkpoints, logs, and evaluation results.

    opts : list[Any]
        List of CLI overrides in ``key=value`` format.
        Example: ``["train.batch_size_per_gpu=64", "optim.lr=0.001"]``

    Example:
    -------
    .. code-block:: python

        args = DinoV3SetupArgs(
            config_file="configs/train/vitl_im1k.yaml",
            output_dir="./outputs/experiment_1",
            opts=["train.batch_size_per_gpu=32", "optim.epochs=50"],
        )
    """

    config_file: str
    pretrained_weights: str | None = None
    shard_unsharded_model: bool = False
    output_dir: str = ""
    opts: list[Any] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        """Convert OmegaConf ListConfig to regular list for serialization compatibility."""
        # When loaded from benchmark.yaml, self.opts is a frozen omegaconf.ListConfig,
        # which works everywhere except when we want to modify it or when
        # we try to json-serialize it. So we convert it to a regular list here.
        if OmegaConf.is_config(self.opts):
            self.opts = list(OmegaConf.to_object(self.opts))  # type: ignore[arg-type]


def apply_scaling_rules_to_cfg(cfg: DictConfig) -> DictConfig:
    """
    Apply learning rate scaling rules based on effective batch size.

    Distributed training typically uses larger effective batch sizes (local batch
    size × world size). To maintain training dynamics, the learning rate should
    be scaled accordingly.

    Supported scaling rules:

    - ``linear_wrt_256``: ``lr *= (batch_size × world_size) / 256``
    - ``sqrt_wrt_1024``: ``lr *= 4 × sqrt((batch_size × world_size) / 1024)``

    Parameters:
    ----------
    cfg : DictConfig
        Configuration object with ``optim.scaling_rule``, ``optim.lr``,
        ``train.batch_size_per_gpu`` fields.

    Returns:
    -------
    DictConfig
        The same config object with ``optim.lr`` modified in-place.

    Raises:
    ------
    AssertionError
        If distributed training is not enabled (needed for world size).

    Note:
    ----
    For configs using ``schedules`` (v2 format), scaling is deferred to
    schedule building and this function returns without modification.
    """
    assert distributed.is_enabled(), "Setup distributed to get global size !"
    if "schedules" in cfg:
        # For schedules v2, the scaling rules are applied when building the schedules, the config is not modified
        return cfg

    if cfg.optim.scaling_rule == "linear_wrt_256":
        old_lr = cfg.optim.lr
        cfg.optim.lr *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(f"linear scaling learning rate; old: {old_lr}, new: {cfg.optim.lr}")
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        old_lr = cfg.optim.lr
        cfg.optim.lr *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; old: {old_lr}, new: {cfg.optim.lr}")
    return cfg


def write_config(cfg: DictConfig, output_dir: str, name: str = "config.yaml") -> str:
    """
    Write configuration to a YAML file.

    Saves the merged configuration for reproducibility and logging.
    Also logs the full config to the console.

    Parameters:
    ----------
    cfg : DictConfig
        Configuration object to save.

    output_dir : str
        Directory to save the config file.

    name : str, default="config.yaml"
        Filename for the saved configuration.

    Returns:
    -------
    str
        Absolute path to the saved configuration file.
    """
    logger.info(OmegaConf.to_yaml(cfg))
    output_dir = os.path.abspath(output_dir)
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_default_config() -> DictConfig:
    """
    Load the default SSL configuration.

    Returns:
    -------
    DictConfig
        Default configuration loaded from ``ssl_default_config.yaml``.
    """
    p = pathlib.Path(__file__).parent / "ssl_default_config.yaml"
    return OmegaConf.load(p)  # type: ignore[return-value]


def get_cfg_from_args(
    args: DinoV3SetupArgs,
    multidistillation: bool = False,
    strict: bool = True,
) -> DictConfig:
    """
    Build merged configuration from arguments.

    Merges configurations in order: default → user config → CLI overrides.
    For multidistillation, skips default config merging.

    Parameters:
    ----------
    args : DinoV3SetupArgs
        Setup arguments containing config file path and overrides.

    multidistillation : bool, default=False
        If True, skip default config merging (used for multi-student distillation).

    strict : bool, default=True
        If True, raise error on unknown config keys (struct mode).

    Returns:
    -------
    DictConfig
        Merged configuration object.
    """
    overrides = [*args.opts]
    if args.output_dir is not None:
        overrides.append(f"train.output_dir={os.path.realpath(args.output_dir)}")

    # Config file
    cfg = OmegaConf.load(args.config_file)

    # Command line overrides
    opts_cfg = OmegaConf.from_cli(overrides)

    if multidistillation:
        cfg = OmegaConf.merge(cfg, opts_cfg)
    else:
        # Default config
        default_cfg = get_default_config()
        if strict:
            OmegaConf.set_struct(default_cfg, True)
        cfg = OmegaConf.merge(default_cfg, cfg, opts_cfg)
    return cfg  # type: ignore[return-value]


def setup_config(args: DinoV3SetupArgs, strict_cfg: bool = True) -> DictConfig:
    """
    Create and setup training configuration.

    This is the main entry point for configuration setup. It:

    1. Loads and merges configurations
    2. Logs the setup arguments
    3. Writes config to output directory
    4. Applies learning rate scaling rules

    Parameters:
    ----------
    args : DinoV3SetupArgs
        Setup arguments containing config file path and overrides.

    strict_cfg : bool, default=True
        If True, raise error on unknown config keys.

    Returns:
    -------
    DictConfig
        Final merged and scaled configuration.
    """
    # Create the cfg with OmegaConf
    cfg = get_cfg_from_args(args, strict=strict_cfg)
    # setup distributed, logging, and random seeds
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # dump config before modifying so it can be reloaded
    if args.output_dir is not None:
        write_config(cfg, args.output_dir)
    # modify the config inplace by applying scaling rules
    apply_scaling_rules_to_cfg(cfg)
    return cfg


def _enumerate_all_subgroup_ranks(
    all_subgroup_rank_spans: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """
    Expand process subgroup rank spans to enumerated rank tuples.

    Converts compact span notation to explicit rank enumeration for
    distributed process group creation.

    Parameters:
    ----------
    all_subgroup_rank_spans : tuple[tuple[int, int], ...]
        Sequence of (first_rank, last_rank) spans, one per subgroup.
        Example: ``((0, 1), (2, 3), (4, 7))``

    Returns:
    -------
    tuple[tuple[int, ...], ...]
        Tuple of rank tuples, one per subgroup.
        Example: ``((0, 1, 2), (3, 4), (5, 6, 7, 8))``

    Raises:
    ------
    AssertionError
        If any span has first > last.
    """
    for first, last in all_subgroup_rank_spans:
        assert first <= last
    return tuple(tuple(range(first, last + 1)) for first, last in all_subgroup_rank_spans)


def setup_multidistillation(args: DinoV3SetupArgs) -> DictConfig:
    """
    Setup configuration for multi-student knowledge distillation.

    Multi-distillation trains multiple student models simultaneously, each on a
    subset of GPUs. This function:

    1. Loads the base multi-distillation config
    2. Creates process subgroups for each student
    3. Assigns the current rank to its student configuration
    4. Merges and scales the configuration

    Architecture:
    ------------
    ::

        World (8 GPUs total)
        ├── Student A (ranks 0-3): ViT-Small
        ├── Student B (ranks 4-5): ViT-Base
        └── Student C (ranks 6-7): ViT-Large

    Parameters:
    ----------
    args : DinoV3SetupArgs
        Setup arguments. The config file should have ``multidistillation.enabled=true``
        and define students with their rank ranges.

    Returns:
    -------
    DictConfig
        Configuration for this rank's student model.

    Raises:
    ------
    AssertionError
        If multidistillation is not enabled or rank is not in any student range.
    """
    base_output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    # get config file for this rank
    base_cfg = OmegaConf.load(args.config_file)
    assert base_cfg.multidistillation.enabled

    global_batch_size = base_cfg.multidistillation.global_batch_size

    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_rank()

    # build process subgroups
    all_subgroup_rank_spans = tuple(
        (student.ranks_range[0], student.ranks_range[1] - 1) for student in base_cfg.multidistillation.students
    )
    all_subgroup_ranks = _enumerate_all_subgroup_ranks(all_subgroup_rank_spans)
    distributed.new_subgroups(all_subgroup_ranks)

    # Find which student this rank belongs to
    current_student = None
    for student in base_cfg.multidistillation.students:
        if rank in range(*student.ranks_range):
            current_student = student
            break
    assert current_student is not None, "rank of worker not in defined range"

    name: str = current_student.name
    config_path: str = current_student.config_path
    n_gpus = current_student.ranks_range[1] - current_student.ranks_range[0]
    assert global_batch_size % n_gpus == 0
    total_n_gpus = distributed.get_world_size()

    args.output_dir = os.path.join(base_output_dir, name)
    args.opts += [f"train.output_dir={args.output_dir}"]
    args.opts += [f"train.batch_size_per_gpu={global_batch_size // total_n_gpus}"]
    args.config_file = os.path.abspath(config_path)
    default_cfg = get_default_config()
    cfg = OmegaConf.load(args.config_file)
    # Merge order: defaults → base multidist config → student config → CLI overrides
    cfg = OmegaConf.merge(default_cfg, base_cfg, cfg, OmegaConf.from_cli(args.opts))

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)

    fix_random_seeds(seed + rank)

    write_config(cfg, args.output_dir)  # type: ignore[arg-type]
    apply_scaling_rules_to_cfg(cfg)  # type: ignore[arg-type]

    return cfg  # type: ignore[return-value]


def setup_job(
    output_dir: str | None = None,
    distributed_enabled: bool = True,
    logging_enabled: bool = True,
    seed: int | None = 0,
    restrict_print_to_main_process: bool = True,
    distributed_timeout: timedelta | None = None,
) -> None:
    """
    Initialize a DINOv3 training or evaluation job.

    This is the standard entry point for any DINOv3 job. It sets up:

    1. **Output directory**: Creates if needed
    2. **Logging**: File and console logging with rank filtering
    3. **Distributed training**: NCCL backend initialization
    4. **Random seeds**: Per-rank seeding for reproducibility
    5. **Environment info**: Logs git SHA, conda env, Python path

    Parameters:
    ----------
    output_dir : str | None, default=None
        Directory for logs and outputs. Created if it doesn't exist.

    distributed_enabled : bool, default=True
        If True, initialize PyTorch distributed with NCCL backend.

    logging_enabled : bool, default=True
        If True, setup file and console logging.

    seed : int | None, default=0
        Base random seed. Actual seed is ``seed + rank`` for per-rank variation.
        If None, random seeding is skipped.

    restrict_print_to_main_process : bool, default=True
        If True, only rank 0 prints to console (others log to file only).

    distributed_timeout : timedelta | None, default=None
        Timeout for distributed operations. If None, uses PyTorch default.

    Example:
    -------
    .. code-block:: python

        from dinov3.configs.config import setup_job, exit_job

        setup_job(
            output_dir="./outputs/experiment",
            seed=42,
            distributed_timeout=timedelta(minutes=30),
        )

        try:
            # Training code here...
            pass
        finally:
            exit_job()
    """
    if output_dir is not None:
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if logging_enabled:
        setup_logging(
            output=output_dir,
            level=logging.INFO,
            log_to_stdout_only_in_main_process=restrict_print_to_main_process,
        )

    if distributed_enabled:
        distributed.enable(
            overwrite=True,
            nccl_async_error_handling=True,
            restrict_print_to_main_process=restrict_print_to_main_process,
            timeout=distributed_timeout,
        )

    if seed is not None:
        rank = distributed.get_rank()
        fix_random_seeds(seed + rank)

    logger = logging.getLogger("dinov3")
    logger.info("git:\n  {}\n".format(get_sha()))

    # Log some python info
    conda_env_name, conda_env_path = get_conda_env()
    logger.info(f"conda env name: {conda_env_name}")
    logger.info(f"conda env path: {conda_env_path}")
    logger.info(f"python path: {sys.path}")


def exit_job(distributed_enabled: bool = True, logging_enabled: bool = True) -> None:
    """
    Clean up resources after a DINOv3 job completes.

    Should be called at the end of training/evaluation to properly shut down
    distributed processes and close log handlers.

    Parameters:
    ----------
    distributed_enabled : bool, default=True
        If True, call ``distributed.disable()`` to clean up process groups.

    logging_enabled : bool, default=True
        If True, call ``cleanup_logging()`` to close file handlers.

    Example:
    -------
    .. code-block:: python

        try:
            # Training code...
            pass
        finally:
            exit_job()
    """
    if distributed_enabled:
        distributed.disable()
    if logging_enabled:
        cleanup_logging()
