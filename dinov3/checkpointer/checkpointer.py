# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Distributed Checkpoint Manager for DINOv3 Training.

This module provides a unified checkpointing system that works seamlessly with:
- Plain PyTorch models
- DistributedDataParallel (DDP) wrapped models
- Fully Sharded Data Parallel (FSDP/FSDP2) wrapped models

Key Features:
------------
1. **Atomic Saves**: Uses temporary directories + rename for crash-safe checkpointing
2. **Rank-Agnostic Loading**: Checkpoints saved on N GPUs can be loaded on M GPUs
3. **Flexible Retention**: Configurable policies (keep all, last, best, etc.)
4. **Selective Saving**: Register hooks to exclude frozen weights from checkpoints
5. **DCP Integration**: Built on PyTorch's Distributed Checkpoint (DCP) for sharded saves

Directory Structure:
-------------------
::

    output_dir/
    ├── ckpt/                      # Checkpoint root directory
    │   ├── 0/                     # Iteration 0 checkpoint (DCP sharded format)
    │   │   ├── .metadata          # DCP metadata file
    │   │   ├── __0_0.distcp       # Sharded tensor data (rank 0)
    │   │   ├── __1_0.distcp       # Sharded tensor data (rank 1)
    │   │   └── ...
    │   ├── 99/                    # Iteration 99 checkpoint
    │   ├── 199/                   # Iteration 199 checkpoint
    │   ├── 199_keep/              # Manually preserved copy (hardlinked)
    │   ├── best/                  # Best validation checkpoint
    │   ├── final/                 # Final training checkpoint
    │   └── ...
    └── eval/
        ├── training_12345/        # Evaluation outputs
        │   └── teacher_checkpoint.pth  # Consolidated teacher weights
        └── ...

Checkpoint Contents:
-------------------
Each DCP checkpoint directory contains:
- ``iteration``: Training iteration number (int or str like "best", "final")
- ``model``: Model state dict (sharded across ranks for FSDP)
- ``optimizer``: Optimizer state dict (optional, sharded for FSDP)
- Additional ``Stateful`` objects passed via ``**others``

Usage Examples:
--------------
**Saving a checkpoint:**

.. code-block:: python

    from dinov3.checkpointer import save_checkpoint

    save_checkpoint(
        ckpt_dir="output/ckpt/100",
        iteration=100,
        model=fsdp_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,  # Any Stateful object
    )

**Loading a checkpoint:**

.. code-block:: python

    from dinov3.checkpointer import load_checkpoint

    iteration = load_checkpoint(
        ckpt_dir="output/ckpt/100",
        model=fsdp_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

**Finding and resuming from latest:**

.. code-block:: python

    from dinov3.checkpointer import find_latest_checkpoint, load_checkpoint

    latest = find_latest_checkpoint("output/ckpt")
    if latest:
        iteration = load_checkpoint(latest, model=model, optimizer=opt)

References:
----------
- PyTorch DCP Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- PyTorch DCP Docs: https://pytorch.org/docs/stable/distributed.checkpoint.html
- FSDP Checkpointing: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
from torch.distributed.checkpoint.stateful import Stateful

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger("dinov3")

# ==============================================================================
# Checkpoint Retention Policy
# ==============================================================================


class CheckpointRetentionPolicy(Enum):
    """
    Enumeration defining checkpoint retention strategies.

    Controls which checkpoints are preserved during training and cleanup.
    Used by :func:`cleanup_checkpoint` and :func:`keep_last_n_checkpoints`.

    Members:
    -------
    ALL
        Keep all periodic checkpoints. No automatic cleanup.
        Use when disk space is not a concern and you want full history.

    BEST
        Keep only the checkpoint with best validation performance.
        The "best" checkpoint must be explicitly saved to a ``best/`` subdirectory.

    LAST
        Keep only the final checkpoint (saved to ``final/`` subdirectory).
        Periodic checkpoints are cleaned up after training completes.

    LAST_AND_BEST
        Keep both ``final/`` and ``best/`` checkpoints.
        Most common choice for production training.

    NONE
        Do not keep any checkpoints after training.
        Useful for hyperparameter sweeps where only metrics matter.

    Example:
    -------
    .. code-block:: python

        policy = CheckpointRetentionPolicy.LAST_AND_BEST
        cleanup_checkpoint("output/ckpt", policy)
        # Keeps: output/ckpt/final/, output/ckpt/best/
        # Deletes: output/ckpt/0/, output/ckpt/100/, etc.
    """

    ALL = "all"  # Keep all checkpoints (no automatic cleanup)
    BEST = "best"  # Keep only best validation checkpoint
    LAST = "last"  # Keep only final checkpoint
    LAST_AND_BEST = "last_and_best"  # Keep both final and best
    NONE = "none"  # Do not keep any checkpoints

    @property
    def keep_filters(self) -> set[str]:
        """
        Directory names that are protected from cleanup.

        Returns:
        -------
        set[str]
            Set of directory names (not paths) that should not be deleted.
            Empty set for ALL policy (nothing is protected because nothing is deleted).
            Empty set for NONE policy (everything is deleted).

        Note:
        ----
        These are exact name matches, not glob patterns. A checkpoint at
        ``output/ckpt/final/`` is protected if "final" is in the filter set.
        """
        if self == CheckpointRetentionPolicy.LAST:
            return {"final"}
        if self == CheckpointRetentionPolicy.BEST:
            return {"best"}
        if self == CheckpointRetentionPolicy.LAST_AND_BEST:
            return {"final", "best"}
        if self == CheckpointRetentionPolicy.ALL:
            return set()  # No filter needed - we keep everything
        return set()  # NONE - no protection, everything deleted

    @property
    def max_to_keep(self) -> int | None:
        """
        Maximum number of periodic (iteration-numbered) checkpoints to keep.

        This controls how many checkpoints like ``0/``, ``100/``, ``200/``
        are kept concurrently. Does not affect named checkpoints like
        ``best/`` or ``final/``.

        Returns:
        -------
        int | None
            Number of periodic checkpoints to retain, or None to keep all.
            - ALL policy: None (keep unlimited periodic checkpoints)
            - Other policies: 1 (keep only the most recent periodic checkpoint)
        """
        if self == CheckpointRetentionPolicy.ALL:
            return None
        return 1


# ==============================================================================
# Core Checkpoint Save/Load Functions
# ==============================================================================


def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    iteration: int | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = True,
    process_group: dist.ProcessGroup | None = None,
    **others: Stateful,
) -> None:
    """
    Save a distributed checkpoint for any PyTorch model type.

    This function provides atomic, crash-safe checkpointing that works with:
    - Plain PyTorch models
    - DDP-wrapped models (``DistributedDataParallel``)
    - FSDP/FSDP2-wrapped models (``FullyShardedDataParallel``)

    The save is atomic: writes to a temporary directory first, then renames
    to the final location. This prevents corrupted checkpoints from partial
    writes during crashes.

    Parameters:
    ----------
    ckpt_dir : str | Path
        Destination directory for the checkpoint. Will be created if it
        doesn't exist. Example: ``"output/ckpt/199"`` for iteration 199.

    iteration : int | str
        Training iteration or named checkpoint identifier.
        - Integer for periodic checkpoints: ``100``, ``200``, etc.
        - String for special checkpoints: ``"best"``, ``"final"``

    model : torch.nn.Module
        The model to checkpoint. Can be plain, DDP-wrapped, or FSDP-wrapped.
        State dict extraction is handled automatically.

    optimizer : torch.optim.Optimizer | None, optional
        Optimizer to save. If None, only model state is saved.
        For FSDP models, optimizer state is automatically sharded.

    overwrite : bool, default=True
        If True, delete existing checkpoint at ``ckpt_dir`` before saving.
        If False, raise RuntimeError if checkpoint already exists.

    process_group : dist.ProcessGroup | None, optional
        Process group for distributed operations. If None, uses the
        default process group (all ranks).

    **others : Stateful
        Additional stateful objects to save (e.g., lr_scheduler, scaler).
        Must implement ``state_dict()`` and ``load_state_dict()`` methods.

    Raises:
    ------
    RuntimeError
        If ``overwrite=False`` and checkpoint already exists.

    Note:
    ----
    All ranks must call this function. Rank 0 coordinates file operations,
    other ranks participate in distributed state dict gathering.

    Example:
    -------
    .. code-block:: python

        # Save periodic checkpoint
        save_checkpoint(
            "output/ckpt/100",
            iteration=100,
            model=fsdp_model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            grad_scaler=scaler,
        )

        # Save best validation checkpoint
        save_checkpoint(
            "output/ckpt/best",
            iteration="best",
            model=fsdp_model,
        )
    """
    rank = torch.distributed.get_rank(group=process_group)

    # Rank 0 checks if the checkpoint directory exists, but all ranks need to know if if exists,
    # so they can raise an error when overwrite is False. If overwrite is True, rank 0 will delete it
    # and other ranks wait for the deletion to finish.
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir_exists = [ckpt_dir.exists() if rank == 0 else None]
    src_rank = 0
    if process_group is not None:
        src_rank = torch.distributed.get_global_rank(group=process_group, group_rank=0)
    torch.distributed.broadcast_object_list(ckpt_dir_exists, src=src_rank, group=process_group)
    ckpt_dir_exists = ckpt_dir_exists[0]
    if ckpt_dir_exists:
        if overwrite:
            if rank == 0:
                if ckpt_dir.is_dir():
                    shutil.rmtree(ckpt_dir)
                else:
                    ckpt_dir.unlink()
                logger.info(f"Deleted: {ckpt_dir}")
            torch.distributed.barrier(group=process_group)
        else:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_dir}")

    # Rank 0 creates a temporary directory for the checkpoint and broadcasts the name to all ranks.
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir_tmp = [tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=ckpt_dir.name) if rank == 0 else None]
    torch.distributed.broadcast_object_list(ckpt_dir_tmp, src=src_rank, group=process_group)
    ckpt_dir_tmp = Path(ckpt_dir_tmp[0])

    to_save = {"iteration": iteration}
    to_save["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_save["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_save.update(others)
    dcp.save(
        to_save,
        storage_writer=dcpfs.FileSystemWriter(ckpt_dir_tmp),
        process_group=process_group,
    )

    # Rank 0 renames the temporary directory to the final checkpoint directory. All ranks wait for the rename.
    if rank == 0:
        ckpt_dir_tmp.rename(ckpt_dir)
    torch.distributed.barrier()

    logger.info(f"Saved: {ckpt_dir}")


def load_checkpoint(
    ckpt_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict_loading: bool = True,
    process_group: dist.ProcessGroup | None = None,
    **others: Stateful,
) -> int | str | None:
    """
    Load a distributed checkpoint into any PyTorch model type.

    This function provides flexible checkpoint loading with automatic
    resharding. Key capabilities:

    - **Rank-agnostic**: Load checkpoint saved on N GPUs to M GPUs
    - **Wrapper-agnostic**: Load regardless of DDP/FSDP wrapping differences
    - **Compile-agnostic**: Works even if ``torch.compile`` status differs
    - **AC-agnostic**: Handles activation checkpointing differences

    Parameters:
    ----------
    ckpt_dir : str | Path
        Directory containing the checkpoint to load.
        Example: ``"output/ckpt/199"`` or ``"output/ckpt/best"``

    model : torch.nn.Module
        Target model to load weights into. Must have compatible architecture
        with the saved checkpoint (same layer names/shapes).

    optimizer : torch.optim.Optimizer | None, optional
        Optimizer to restore state into. If None, optimizer state is skipped.
        The optimizer must be constructed with the same param groups as when
        the checkpoint was saved.

    strict_loading : bool, default=True
        If True, require exact match between checkpoint and model keys.
        If False, allow partial loading (missing/extra keys are ignored).
        Useful for transfer learning or architecture modifications.

    process_group : dist.ProcessGroup | None, optional
        Process group for distributed operations. If None, uses default.

    **others : Stateful
        Additional stateful objects to restore (must match what was saved).

    Returns:
    -------
    int | str | None
        The iteration value stored in the checkpoint.
        Returns None if iteration was not saved or loading failed.

    Note:
    ----
    All ranks must call this function. The DCP library handles automatic
    resharding of sharded tensors across the new process group topology.

    Example:
    -------
    .. code-block:: python

        # Resume training from latest checkpoint
        latest_ckpt = find_latest_checkpoint("output/ckpt")
        if latest_ckpt:
            start_iteration = load_checkpoint(
                latest_ckpt,
                model=fsdp_model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
            )
        else:
            start_iteration = 0

        # Load with partial matching (e.g., new head layer)
        load_checkpoint(
            "pretrained/ckpt/final",
            model=model_with_new_head,
            strict_loading=False,
        )
    """
    ckpt_dir = Path(ckpt_dir)
    to_load = {"iteration": None}
    to_load["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_load["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_load.update(others)
    dcp.load(
        to_load,
        storage_reader=dcpfs.FileSystemReader(ckpt_dir),
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=not strict_loading),
        process_group=process_group,
    )
    iteration = to_load["iteration"]
    dcpsd.set_model_state_dict(model, to_load["model"])
    if optimizer is not None:
        dcpsd.set_optimizer_state_dict(model, optimizer, to_load["optimizer"])
    logger.info(f"Loaded: {ckpt_dir}")
    return iteration


# ==============================================================================
# Selective Saving Hooks
# ==============================================================================


def register_dont_save_hooks(module: torch.nn.Module, dont_save: Sequence[str]) -> None:
    """
    Register hooks to exclude specific weights from checkpoint saves.

    This is useful when a model contains frozen pretrained weights that
    don't need to be saved repeatedly. By excluding them, checkpoint size
    is reduced and save/load times improve.

    The hooks work in both directions:

    1. **Save hook**: Removes specified keys from state dict before saving
    2. **Load hooks**: Suppresses "missing key" errors for excluded weights

    Parameters:
    ----------
    module : torch.nn.Module
        The module to register hooks on. Typically the top-level model.

    dont_save : Sequence[str]
        List of parameter names to exclude from checkpoints.
        Names should be relative to the module (e.g., ``"backbone.layer1.weight"``).

        Note: Activation checkpointing may add ``_checkpoint_wrapped_module.``
        prefix to names. This function handles that automatically.

    Example:
    -------
    .. code-block:: python

        # Model with frozen backbone + trainable head
        model = nn.Sequential(
            backbone,  # Frozen, loaded from torch hub
            head,      # Trainable
        )

        # Get backbone parameter names
        backbone_params = [n for n, _ in backbone.named_parameters()]
        # e.g., ['0.conv1.weight', '0.bn1.weight', ...]

        # Register hooks to exclude backbone from checkpoints
        register_dont_save_hooks(model, backbone_params)

        # Now save_checkpoint will only save head weights
        save_checkpoint("output/ckpt/0", iteration=0, model=model)

    Warning:
    -------
    When loading a checkpoint saved with excluded weights, ensure the
    frozen weights are loaded separately (e.g., from torch hub or a
    pretrained checkpoint file).

    See Also:
    --------
    - ``torch.nn.Module.register_state_dict_post_hook``
    - ``torch.nn.Module.register_load_state_dict_pre_hook``
    """

    def state_dict_post_hook(module, state_dict, prefix, local_metadata):
        # Remove frozen weights so they won't get saved.
        # If this module is not the top-level module, its weights will have a prefix in the state dict.
        nonlocal _dont_save
        for k in _dont_save:
            del state_dict[prefix + k]

    def load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # This pre hook exists only to pass the prefix to the post hook when loading the state dict.
        nonlocal _prefix
        assert _prefix is None
        _prefix = prefix

    def load_state_dict_post_hook(module, incompatible_keys):
        # Remove the frozen weights from the missing keys so they don't raise an error.
        nonlocal _prefix
        assert _prefix is not None
        to_remove = []
        for missing_key in incompatible_keys.missing_keys:
            k = missing_key.removeprefix(_prefix)
            k = k.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing
            if k in _dont_save:
                to_remove.append(missing_key)
        for r in to_remove:
            incompatible_keys.missing_keys.remove(r)
        _prefix = None

    _dont_save = set(name.replace("_checkpoint_wrapped_module.", "") for name in dont_save)
    _prefix = None
    module.register_state_dict_post_hook(state_dict_post_hook)
    module.register_load_state_dict_pre_hook(load_state_dict_pre_hook)
    module.register_load_state_dict_post_hook(load_state_dict_post_hook)


# ==============================================================================
# Checkpoint Discovery and Management
# ==============================================================================


def find_all_checkpoints(ckpt_dir: Path | str) -> list[Path]:
    """
    Find all periodic (iteration-numbered) checkpoints in a directory.

    Searches for subdirectories with integer names (e.g., ``0/``, ``100/``,
    ``200/``). Ignores named checkpoints like ``best/`` or ``final/``.

    Parameters:
    ----------
    ckpt_dir : Path | str
        Root checkpoint directory to search.

    Returns:
    -------
    list[Path]
        Sorted list of checkpoint paths, from lowest to highest iteration.
        Empty list if directory doesn't exist or has no valid checkpoints.

    Example:
    -------
    .. code-block:: python

        # Directory structure:
        # output/ckpt/
        # ├── 0/
        # ├── 100/
        # ├── 200/
        # ├── best/
        # └── final/

        checkpoints = find_all_checkpoints("output/ckpt")
        # Returns: [Path('output/ckpt/0'), Path('output/ckpt/100'), Path('output/ckpt/200')]
        # Note: 'best/' and 'final/' are NOT included
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir() and _is_int(p.name)]
    checkpoints.sort(key=lambda p: int(p.name))
    return checkpoints


def find_latest_checkpoint(ckpt_dir: Path | str) -> Path | None:
    """
    Find the most recent periodic checkpoint by iteration number.

    Useful for implementing training resumption - find where training
    left off and continue from there.

    Parameters:
    ----------
    ckpt_dir : Path | str
        Root checkpoint directory to search.

    Returns:
    -------
    Path | None
        Path to the checkpoint with highest iteration number,
        or None if no valid checkpoints exist.

    Example:
    -------
    .. code-block:: python

        latest = find_latest_checkpoint("output/ckpt")
        if latest:
            print(f"Resuming from {latest.name}")  # e.g., "Resuming from 200"
            iteration = load_checkpoint(latest, model=model)
        else:
            print("Starting fresh training")
            iteration = 0
    """
    checkpoints = find_all_checkpoints(ckpt_dir)
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def keep_last_n_checkpoints(ckpt_dir: Path | str, n: int | None) -> None:
    """
    Retain only the N most recent periodic checkpoints, delete older ones.

    This is the primary mechanism for checkpoint cleanup during training.
    Call after each checkpoint save to maintain disk space.

    Parameters:
    ----------
    ckpt_dir : Path | str
        Root checkpoint directory containing numbered subdirectories.

    n : int | None
        Number of checkpoints to keep. If None, all checkpoints are kept.
        - ``n=1``: Keep only the latest checkpoint
        - ``n=3``: Keep the 3 most recent checkpoints
        - ``n=None``: Keep all checkpoints (no cleanup)

    Note:
    ----
    Only affects periodic (integer-named) checkpoints. Named checkpoints
    like ``best/`` and ``final/`` are not touched by this function.

    Example:
    -------
    .. code-block:: python

        # After saving checkpoint 300
        save_checkpoint("output/ckpt/300", iteration=300, model=model)

        # Keep only the last 2 periodic checkpoints
        keep_last_n_checkpoints("output/ckpt", n=2)
        # Deletes: 0/, 100/
        # Keeps: 200/, 300/, best/, final/
    """
    if n is None:
        return
    checkpoints = find_all_checkpoints(ckpt_dir)
    for ckpt_dir in checkpoints[:-n]:
        try:
            shutil.rmtree(ckpt_dir)
            logger.info(f"Deleted: {ckpt_dir}")
        except Exception:
            logger.exception(f"Failed to delete: {ckpt_dir}")


def keep_checkpoint_copy(src: Path | str) -> None:
    """
    Create a preserved copy of a checkpoint with ``_keep`` suffix.

    Uses hard links (not copies) so disk usage is minimal - the same
    physical data blocks are referenced by both paths. The ``_keep``
    copy survives even if the original is later deleted by cleanup.

    Parameters:
    ----------
    src : Path | str
        Path to the checkpoint directory to preserve.

    Note:
    ----
    Requires a Unix-like filesystem that supports hard links.
    Uses ``cp --recursive --link`` under the hood.

    Example:
    -------
    .. code-block:: python

        # Preserve checkpoint 199 before cleanup might delete it
        keep_checkpoint_copy("output/ckpt/199")
        # Creates: output/ckpt/199_keep/ (hardlinked to 199/)

        # Later, even if 199/ is deleted:
        keep_last_n_checkpoints("output/ckpt", n=1)
        # 199/ is deleted, but 199_keep/ still exists with all data
    """
    src = Path(src)
    dst = src.parent / f"{src.name}_keep"
    subprocess.check_output(["cp", "--recursive", "--link", src, dst])
    logger.info(f"Copied: {src} -> {dst}")


def _is_int(s: str) -> bool:
    """
    Check if a string represents a valid integer.

    Used to distinguish periodic checkpoints (integer names like "100")
    from named checkpoints (string names like "best" or "final").

    Parameters:
    ----------
    s : str
        String to check.

    Returns:
    -------
    bool
        True if string can be parsed as an integer, False otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


# ==============================================================================
# Model Initialization from Checkpoints
# ==============================================================================


def init_fsdp_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    skip_load_keys: list[str] | None = None,
    keys_not_sharded: list[str] | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """
    Initialize an FSDP2 model from either DCP or standard PyTorch checkpoint.

    This function handles two checkpoint formats:

    1. **DCP Format** (directory): Uses :func:`load_checkpoint` for distributed loading
    2. **Standard Format** (.pth file): Loads on CPU, then distributes to FSDP shards

    For standard checkpoints, the function:
    - Extracts the "teacher" key from the checkpoint dict
    - Distributes tensors across the device mesh using DTensor
    - Handles mixed sharded/non-sharded parameters via ``keys_not_sharded``

    Parameters:
    ----------
    model : torch.nn.Module
        FSDP2-wrapped model to initialize. Should already be wrapped
        with ``fully_shard()`` before calling this function.

    checkpoint_path : str
        Path to checkpoint. Can be:
        - Directory: Treated as DCP format
        - File: Treated as standard PyTorch checkpoint

    skip_load_keys : list[str] | None, optional
        Parameter name patterns to skip loading. Useful when the checkpoint
        has parameters that don't exist in the model (e.g., head weights
        when doing backbone-only initialization).

    keys_not_sharded : list[str] | None, optional
        Parameter name patterns that should NOT be distributed across
        the device mesh. These are typically small parameters like
        normalization layers or biases that don't benefit from sharding.

    process_group : dist.ProcessGroup | None, optional
        Process group for distributed operations. If None, uses default.

    Example:
    -------
    .. code-block:: python

        # Wrap model with FSDP2
        model = fully_shard(model, ...)

        # Initialize from standard checkpoint (e.g., downloaded weights)
        init_fsdp_model_from_checkpoint(
            model,
            checkpoint_path="dinov2_vitl14_pretrain.pth",
            skip_load_keys=["head"],  # Skip classification head
            keys_not_sharded=["norm", "bias"],  # Don't shard small params
        )

        # Or initialize from DCP checkpoint (from previous training)
        init_fsdp_model_from_checkpoint(
            model,
            checkpoint_path="output/ckpt/final",  # Directory
        )
    """
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        chkpt = torch.load(checkpoint_path, map_location="cpu")["teacher"]
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        if process_group is None:
            world_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dist.get_world_size(),),
                mesh_dim_names=("dp",),
            )
        else:
            world_mesh = DeviceMesh.from_group(process_group, "cuda")
        chkpt = {
            key: (
                torch.distributed.tensor.distribute_tensor(tensor, world_mesh, src_data_rank=None)
                if not any(key_not_sharded in key for key_not_sharded in keys_not_sharded)
                else tensor
            )
            for key, tensor in chkpt.items()
        }
        model.load_state_dict(
            {
                key: tensor
                for key, tensor in chkpt.items()
                if not any(skip_load_key in key for skip_load_key in skip_load_keys)
            }
        )
    else:  # DCP checkpoint
        load_checkpoint(ckpt_dir=checkpoint_path, model=model, process_group=process_group)


def init_model_from_checkpoint_for_evals(
    model: torch.nn.Module,
    pretrained_weights: str | Path,
    checkpoint_key: str | None = None,
) -> None:
    """
    Initialize a non-distributed model for evaluation from a PyTorch checkpoint.

    This is the simplest loading function, designed for single-GPU evaluation.
    It handles common checkpoint variations:

    - Nested state dicts (extracts specified key)
    - DDP prefix removal (``module.`` prefix)
    - Multicrop wrapper prefix removal (``backbone.`` prefix)

    Parameters:
    ----------
    model : torch.nn.Module
        Target model to load weights into. Should NOT be wrapped with
        DDP or FSDP - this is for plain evaluation models.

    pretrained_weights : str | Path
        Path to ``.pth`` checkpoint file.

    checkpoint_key : str | None, optional
        Key to extract from checkpoint dict. Common values:
        - ``"teacher"``: For DINOv2/v3 teacher weights
        - ``"student"``: For DINOv2/v3 student weights
        - ``"model"``: Generic model state dict
        - ``None``: Use entire checkpoint as state dict

    Note:
    ----
    Uses ``strict=False`` for ``load_state_dict()``, so missing or
    unexpected keys are logged but don't raise errors.

    Example:
    -------
    .. code-block:: python

        # Load teacher weights for evaluation
        model = vit_large(patch_size=14)
        init_model_from_checkpoint_for_evals(
            model,
            pretrained_weights="output/eval/training_12345/teacher_checkpoint.pth",
            checkpoint_key="teacher",
        )

        # Evaluate
        model.eval()
        with torch.no_grad():
            features = model(images)
    """
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def cleanup_checkpoint(
    ckpt_dir: str | Path,
    checkpoint_retention_policy: CheckpointRetentionPolicy,
) -> None:
    """
    Clean up checkpoints according to the specified retention policy.

    Deletes checkpoint directories that don't match the retention policy's
    keep filters. Typically called at the end of training to free disk space.

    Parameters:
    ----------
    ckpt_dir : str | Path
        Root checkpoint directory containing individual checkpoint subdirs.

    checkpoint_retention_policy : CheckpointRetentionPolicy
        Policy determining which checkpoints to keep:

        - ``LAST``: Keep only ``final/``
        - ``BEST``: Keep only ``best/``
        - ``LAST_AND_BEST``: Keep ``final/`` and ``best/``
        - ``ALL``: Keep everything (no cleanup)
        - ``NONE``: Delete everything

    Directory Structure:
    -------------------
    ::

        ckpt_dir/
        ├── 0/       # Periodic checkpoint → MAY be deleted
        ├── 99/      # Periodic checkpoint → MAY be deleted
        ├── 199/     # Periodic checkpoint → MAY be deleted
        ├── 299/     # Periodic checkpoint → MAY be deleted
        ├── best/    # Best validation    → Protected by BEST, LAST_AND_BEST
        └── final/   # Final checkpoint   → Protected by LAST, LAST_AND_BEST

    Example:
    -------
    .. code-block:: python

        # At end of training, clean up keeping only final and best
        cleanup_checkpoint(
            "output/ckpt",
            CheckpointRetentionPolicy.LAST_AND_BEST,
        )
        # Deletes: 0/, 99/, 199/, 299/
        # Keeps: best/, final/

    Warning:
    -------
    This operation is irreversible. Use :func:`keep_checkpoint_copy` to
    preserve important checkpoints before cleanup.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoint_filters = checkpoint_retention_policy.keep_filters
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir()]
    for checkpoint in checkpoints:
        if checkpoint in checkpoint_filters:
            continue
        try:
            shutil.rmtree(checkpoint)
            logger.info(f"Deleted: {checkpoint}")
        except Exception:
            logger.exception(f"Failed to delete: {checkpoint}")
