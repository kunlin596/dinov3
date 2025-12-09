# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Self-Supervised Learning Meta-Architecture for DINOv3.

This module implements the core training architecture for DINOv3 self-supervised
learning, combining multiple loss functions (DINO, iBOT, KoLeo, Gram) in a
teacher-student framework with exponential moving average (EMA) updates.

Architecture Overview:
---------------------
::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         SSLMetaArch                                     │
    │                                                                         │
    │  ┌─────────────────┐              ┌─────────────────┐                   │
    │  │    STUDENT      │              │    TEACHER      │                   │
    │  │  (trainable)    │    EMA       │  (frozen, EMA)  │                   │
    │  │                 │ ──────────>  │                 │                   │
    │  │  ├─ backbone    │   update     │  ├─ backbone    │                   │
    │  │  ├─ dino_head   │              │  ├─ dino_head   │                   │
    │  │  └─ ibot_head   │              │  └─ ibot_head   │                   │
    │  └────────┬────────┘              └────────┬────────┘                   │
    │           │                                │                            │
    │           ▼                                ▼                            │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │                        LOSS COMPUTATION                         │    │
    │  │                                                                 │    │
    │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │    │
    │  │  │   DINO   │  │   iBOT   │  │  KoLeo   │  │   Gram   │        │    │
    │  │  │  (CLS)   │  │ (patch)  │  │  (reg)   │  │ (patch)  │        │    │
    │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │    │
    │  │       │              │             │             │              │    │
    │  │       └──────────────┴─────────────┴─────────────┘              │    │
    │  │                              │                                  │    │
    │  │                              ▼                                  │    │
    │  │                     Total Weighted Loss                         │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘

Data Flow:
---------
::

    Input Images
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │            DataAugmentationDINO (Multi-crop)            │
    │                                                         │
    │   ┌───────────────┐        ┌───────────────────────┐    │
    │   │ Global Crops  │        │    Local Crops        │    │
    │   │   (2x 224²)   │        │  (n_local x 96²)      │    │
    │   │ + iBOT masks  │        │  (no masking)         │    │
    │   └───────┬───────┘        └───────────┬───────────┘    │
    └───────────┼────────────────────────────┼────────────────┘
                │                            │
                ▼                            ▼
    ┌───────────────────────────────────────────────────────────┐
    │                    Student Backbone                       │
    │                                                           │
    │   Global crops ──> CLS token + Patch tokens (masked)      │
    │   Local crops  ──> CLS token + Patch tokens               │
    └───────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────┐
    │                    Teacher Backbone                       │
    │                    (no gradients)                         │
    │                                                           │
    │   Global crops ──> CLS token + Patch tokens               │
    │   (Sinkhorn-Knopp centering applied to outputs)           │
    └───────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────┐
    │                    Loss Computation                       │
    │                                                           │
    │   DINO Loss:  Cross-entropy on CLS tokens                 │
    │   iBOT Loss:  Cross-entropy on masked patch tokens        │
    │   KoLeo Loss: Uniformity regularization on CLS features   │
    │   Gram Loss:  Feature correlation matching (optional)     │
    └───────────────────────────────────────────────────────────┘

Loss Functions:
--------------
1. **DINO Loss** (``dino.loss_weight``)
   - Compares student and teacher CLS token distributions
   - Applied to both global-global and local-global crop pairs
   - Uses cross-entropy with soft targets from teacher

2. **iBOT Loss** (``ibot.loss_weight``)
   - Masked Image Modeling (MIM) objective
   - Student predicts teacher's patch tokens for masked regions
   - Only applied to global crops with random masking

3. **KoLeo Loss** (``dino.koleo_loss_weight``)
   - Uniformity regularization on feature embeddings
   - Encourages features to span the hypersphere uniformly
   - Can be computed locally or distributed across GPUs

4. **Gram Loss** (``gram.loss_weight``, optional)
   - Matches feature correlation matrices between student and teacher
   - Uses a separate (possibly frozen) teacher backbone
   - Helps preserve structural information from pretrained models

Key Configuration Options:
-------------------------
- ``crops.local_crops_number``: Number of local crops (typically 8-10)
- ``crops.global_crops_size``: Size of global crops (typically 224)
- ``ibot.mask_ratio_min_max``: Range of masking ratios for iBOT
- ``dino.head_n_prototypes``: Output dimension of DINO head (typically 65536)
- ``gram.use_loss``: Whether to enable Gram loss regularization

Usage Example:
-------------
.. code-block:: python

    from dinov3.train.ssl_meta_arch import SSLMetaArch
    from omegaconf import OmegaConf

    # Load configuration
    cfg = OmegaConf.load("dinov3/configs/train/vitl_im1k_lin834.yaml")

    # Build meta-architecture
    model = SSLMetaArch(cfg)
    model.init_weights()
    model.prepare_for_distributed_training()

    # Training loop
    for data in dataloader:
        loss, metrics = model.forward_backward(
            data,
            teacher_temp=0.04,
            iteration=step,
        )
        optimizer.step()
        model.update_ema(momentum=0.996)

See Also:
--------
- ``dinov3/train/multidist_meta_arch.py``: Multi-student distillation variant
- ``dinov3/loss/``: Individual loss function implementations
- ``dinov3/models/``: Backbone model definitions (ViT variants)
"""

from __future__ import annotations

import gc
import logging
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.configs import get_default_config
from dinov3.data import DataAugmentationDINO
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss, GramLoss, KoLeoLoss, KoLeoLossDistributed, iBOTPatchLoss
from dinov3.models import build_model_from_cfg
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from dinov3.utils import count_parameters

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger("dinov3")


# ==============================================================================


# ==============================================================================


class SSLMetaArch(nn.Module):
    """
    Self-Supervised Learning Meta-Architecture for DINOv3 pretraining.

    This class orchestrates the complete DINOv3 training pipeline, including:

    - Student and teacher network management (with EMA updates)
    - Multi-crop data augmentation coordination
    - Forward pass through student and teacher networks
    - Loss computation (DINO + iBOT + KoLeo + optional Gram)
    - Backward pass and gradient accumulation

    The architecture follows the teacher-student paradigm where:

    1. **Student**: Trained with gradients, processes masked global crops and local crops
    2. **Teacher**: Frozen copy updated via EMA, provides soft targets
    3. **Gram Teacher** (optional): Separate frozen backbone for Gram loss

    Attributes:
    ----------
    cfg : DictConfig
        Complete OmegaConf configuration object.

    student : nn.ModuleDict
        Trainable student model containing:
        - ``backbone``: Vision Transformer backbone
        - ``dino_head``: Projection head for DINO loss (CLS token)
        - ``ibot_head``: Projection head for iBOT loss (patch tokens)

    teacher : nn.ModuleDict
        Frozen teacher model (same structure as student).
        Updated via EMA in :meth:`update_ema`.

    model_ema : nn.ModuleDict
        Alias to teacher (or distillation teacher if enabled).
        This is the model used for EMA target generation.

    gram_teacher : nn.ModuleDict | None
        Optional separate teacher for Gram loss.
        Only created if ``cfg.gram.use_loss=True`` and ``cfg.gram.ema_teacher=False``.

    dino_loss : DINOLoss
        DINO loss module with Sinkhorn-Knopp centering.

    ibot_patch_loss : iBOTPatchLoss
        iBOT masked patch prediction loss.

    koleo_loss : KoLeoLoss | KoLeoLossDistributed
        Uniformity regularization loss.

    gram_loss : GramLoss | None
        Optional Gram matrix matching loss.

    embed_dim : int
        Embedding dimension of the backbone (D).

    dino_out_dim : int
        Output dimension of DINO head (K, number of prototypes).

    Example:
    -------
    .. code-block:: python

        # Standard DINOv3 pretraining setup
        model = SSLMetaArch(cfg)
        model.init_weights()  # Initialize or load pretrained weights
        model.prepare_for_distributed_training()  # Apply FSDP wrapping

        # Build optimizer on student parameters only
        optimizer = torch.optim.AdamW(model.get_params_groups())

        # Training step
        loss, metrics = model.forward_backward(
            data=batch,
            teacher_temp=0.04,
            iteration=current_iteration,
        )
        optimizer.step()
        optimizer.zero_grad()

        # Update teacher via EMA
        model.update_ema(m=0.996)  # m is momentum, higher = slower update

    Note:
    ----
    This class assumes FSDP (Fully Sharded Data Parallel) training with
    the ``SHARD_GRAD_OP`` sharding strategy. The ``prepare_for_distributed_training``
    method must be called before training to apply FSDP wrapping.

    See Also:
    --------
    - :class:`MultiDistillationMetaArch`: For multi-student knowledge distillation
    - :func:`build_model_from_cfg`: Backbone construction utility
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the SSL Meta-Architecture.

        Constructs student, teacher, and optional gram teacher networks along
        with all loss modules. Does NOT initialize weights - call :meth:`init_weights`
        after construction.

        Parameters:
        ----------
        cfg : DictConfig
            Complete configuration object. Key sections:

            - ``student``: Backbone architecture (arch, patch_size, etc.)
            - ``crops``: Data augmentation settings
            - ``dino``: DINO loss configuration
            - ``ibot``: iBOT loss configuration
            - ``gram``: Gram loss configuration (optional)
            - ``distillation``: Knowledge distillation settings
            - ``optim``: Optimizer and scheduler settings
            - ``train``: Training loop settings

        Raises:
        ------
        AssertionError
            If configuration constraints are violated:
            - ``crops.local_crops_number`` must be > 0
            - ``ibot.separate_head`` must be True
            - ``train.centering`` must be "sinkhorn_knopp"
            - ``compute_precision.sharding_strategy`` must be "SHARD_GRAD_OP"

        ValueError
            If Gram loss configuration is inconsistent.
        """
        super().__init__()

        # ==========================================================================
        # Configuration Validation
        # ==========================================================================
        # assert cfg.multidistillation.enabled is False
        assert cfg.crops.local_crops_number > 0
        assert cfg.ibot.separate_head is True
        assert cfg.train.centering == "sinkhorn_knopp"

        # For some reason FULL_SHARD doesn't work
        assert cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"

        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()
        gram_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        gram_backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"Number of parameters: {count_parameters(student_backbone)}")
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        gram_model_dict["backbone"] = gram_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim  # D
        self.dino_out_dim = cfg.dino.head_n_prototypes  # K

        logger.info("OPTIONS -- DINO")
        logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        logger.info(f"OPTIONS -- DINO -- global_ignore_diagonal: {cfg.dino.global_ignore_diagonal}")
        logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        logger.info(f"OPTIONS -- DINO -- head_norm_last_layer: {cfg.dino.head_norm_last_layer}")
        dino_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head_class()
        teacher_model_dict["dino_head"] = dino_head_class()
        self.dino_loss = DINOLoss(self.dino_out_dim)

        logger.info("OPTIONS -- KOLEO")
        logger.info(f"OPTIONS -- KOLEO -- loss_weight: {cfg.dino.koleo_loss_weight}")
        logger.info(f"OPTIONS -- KOLEO -- distributed: {cfg.dino.koleo_loss_distributed}")
        if cfg.dino.koleo_loss_distributed:
            logger.info(f"OPTIONS -- KOLEO -- topk: {cfg.dino.koleo_topk}")
            logger.info(
                f"OPTIONS -- KOLEO -- distributed_loss_group_size: {cfg.dino.koleo_distributed_loss_group_size}"
            )
            assert (
                cfg.dino.koleo_distributed_replicas == 0
            ), "Option `dino.koleo_distributed_replicas` is no longer supported"
            self.koleo_loss = KoLeoLossDistributed(
                topk=cfg.dino.koleo_topk,
                loss_group_size=cfg.dino.koleo_distributed_loss_group_size,
            )
        else:
            assert cfg.dino.koleo_topk == 1, "Non-distributed KoLeo loss only supports `dino.koleo_topk=1`"
            self.koleo_loss = KoLeoLoss()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")

        assert (
            0 <= cfg.ibot.mask_ratio_min_max[0] < cfg.ibot.mask_ratio_min_max[1] <= 1
        ), "provide a valid cfg.ibot.mask_ratio_min_max"
        assert 0 <= cfg.ibot.mask_sample_probability <= 1, "provide a positive mask probability for ibot"
        logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
        logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_norm_last_layer: {cfg.ibot.head_norm_last_layer}")
        ibot_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
        student_model_dict["ibot_head"] = ibot_head_class()
        teacher_model_dict["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes)

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.model_ema = self.teacher  # this may be overwritten for distillation
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        if cfg.distillation.enabled:
            self._setup_distillation()
        # No grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.ema_params_lists = None

        # getting config params fixed:
        self.n_local_crops = self.cfg.crops.local_crops_number
        self.is_distillation_enabled = self.cfg.distillation.enabled
        self.dino_global_ignore_diagonal = self.cfg.dino.global_ignore_diagonal
        self.dino_loss_weight = self.cfg.dino.loss_weight
        self.dino_koleo_loss_weight = self.cfg.dino.koleo_loss_weight
        self.ibot_loss_weight = self.cfg.ibot.loss_weight

        # Local loss reweighting
        if self.cfg.dino.reweight_dino_local_loss:
            iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
            total_iterations = iter_per_epoch * cfg.optim.epochs
            schedule_cfg = cfg.dino.local_loss_weight_schedule
            self.dino_local_loss_schedule = linear_warmup_cosine_decay(
                start=schedule_cfg.start,
                peak=schedule_cfg.peak,
                end=schedule_cfg.end,
                warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                total_iterations=total_iterations,
                cosine_iterations=(
                    iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                ),
            )

        # Gram
        self.gram_use_loss = self.cfg.gram.use_loss
        self.gram_ema_teacher = False
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        if self.gram_use_loss:
            # Gram regularization
            self.gram_loss = GramLoss(
                apply_norm=self.cfg.gram.normalized,
                remove_only_teacher_neg=self.cfg.gram.remove_only_teacher_neg,
                remove_neg=self.cfg.gram.remove_neg,
            )
            # Construct gram teacher
            self.has_gram_teacher = True if not cfg.gram.ema_teacher else False
            if self.has_gram_teacher:
                self.gram_teacher = nn.ModuleDict(gram_model_dict)
                self.gram_teacher.requires_grad_(False)
                logger.info(f"Gram teacher parameter at init: {next(self.gram_teacher.named_parameters())}")
            else:
                self.gram_teacher = None

            self.gram_loss_weight = self.cfg.gram.loss_weight
            if self.cfg.gram.get("loss_weight_schedule"):
                iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
                total_iterations = iter_per_epoch * cfg.optim.epochs
                schedule_cfg = self.cfg.gram.loss_weight_schedule
                self.gram_loss_schedule = linear_warmup_cosine_decay(
                    start=schedule_cfg.start,
                    peak=schedule_cfg.peak,
                    end=schedule_cfg.end,
                    warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                    total_iterations=total_iterations,
                    cosine_iterations=(
                        iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                    ),
                )
                logger.info(f"Applying gram loss weight schedule instead of `cfg.gram.loss_weight`: {schedule_cfg}")
            else:
                self.gram_loss_schedule = None
            self.gram_ema_teacher = self.cfg.gram.ema_teacher  # If true use the EMA_teacher as gram_teacher
            self.gram_ckpt = self.cfg.gram.ckpt  # Checkpoint to the first gram teacher model
            self.gram_img_level = self.cfg.gram.img_level  # Apply the loss on the image, if false on the batch
            self.gram_tokens_used = self.cfg.gram.tokens_used  # Any value in ["all", "masked", "unmasked"]
            # Update the teacher frequently
            self.gram_rep_update = self.cfg.gram.rep_update  # bool, if yes the gram teacher will be updated at the freq
            self.gram_update_frequency = self.cfg.gram.update_frequency  # defined by this var update_frequency
            self.gram_it_first_update = self.cfg.gram.it_first_update  # after iteration it_first_update is passed.
            self.gram_it_load_ema_teacher = (
                self.cfg.gram.it_load_ema_teacher
            )  # after iteration it_load_ema the ema teacher is loaded into the gram teacher
            self.gram_compute_stats = self.cfg.gram.compute_stats  # whether to compute auxiliary stats
            self.gram_params_lists = None

            if self.gram_ema_teacher and self.gram_ckpt is not None:
                raise ValueError(
                    "Cannot use both `gram.ema_teacher` and `gram.ckpt` at the same time. Please set one of them to False."
                )
            if self.gram_ckpt is None and self.gram_it_load_ema_teacher < 0:
                raise ValueError(
                    "If no gram checkpoint is provided, `gram.it_load_ema_teacher` must be set to a non-negative value."
                )

            assert not (self.gram_ema_teacher and self.gram_rep_update)
            assert self.gram_tokens_used in ["all", "masked", "unmasked"]
            # Currently using masked/unmasked not handle at the image-level
            if self.gram_tokens_used in ["masked", "unmasked"]:
                assert self.gram_img_level is False

            logger.info("OPTIONS -- GRAM")
            logger.info(f"OPTIONS -- GRAM -- loss_weight: {cfg.gram.loss_weight}")
            logger.info(f"OPTIONS -- GRAM -- ema teacher: {cfg.gram.ema_teacher}")
            logger.info(f"OPTIONS -- GRAM -- ckpt: {cfg.gram.ckpt}")
            if self.cfg.gram.rep_update:
                logger.info(f"OPTIONS -- GRAM -- repeated update: {cfg.gram.rep_update}")
                logger.info(f"OPTIONS -- GRAM -- update freq: {cfg.gram.update_frequency}")
                logger.info(f"OPTIONS -- GRAM -- iteration first update: {cfg.gram.it_first_update}")

            logger.info(f"OPTIONS -- GRAM -- tokens_used: {cfg.gram.tokens_used}")
            logger.info(f"OPTIONS -- GRAM -- apply normalization: {cfg.gram.normalized}")
            logger.info(f"OPTIONS -- GRAM -- img_level: {cfg.gram.img_level}")
            logger.info(f"OPTIONS -- GRAM -- remove_neg: {cfg.gram.remove_neg}")
            logger.info(f"OPTIONS -- GRAM -- remove_only_teacher_neg: {cfg.gram.remove_only_teacher_neg}")

            if cfg.crops.gram_teacher_crops_size is None and self.has_gram_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size must be set to use gram loss")
            if cfg.crops.gram_teacher_crops_size is not None and self.gram_ema_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size shoud be None when gram.ema_teacher=True")

            self.student_crop_size = cfg.crops.global_crops_size
            self.gram_global_teacher_resize_method = cfg.gram.global_teacher_resize_method
            self.gram_global_teacher_resize_antialias = cfg.gram.global_teacher_resize_antialias
            logger.info(f"OPTIONS -- global crops student/teacher size: {self.student_crop_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher size: {cfg.crops.gram_teacher_crops_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher resize method: {cfg.gram.global_teacher_resize_method}")
            logger.info(
                f"OPTIONS -- global crops GRAM teacher resize antialias: {cfg.gram.global_teacher_resize_antialias}"
            )

    def _setup_distillation(self) -> None:
        """
        Set up knowledge distillation from a larger teacher model.

        When ``cfg.distillation.enabled=True``, this method replaces the default
        teacher (EMA copy of student) with a separate, potentially larger model
        loaded from a different configuration and checkpoint.

        The distillation teacher:
        - Uses its own architecture (can be different size than student)
        - Is loaded from ``cfg.distillation.checkpoint_path``
        - Remains frozen during training
        - Must have matching head dimensions (prototypes) with student

        This enables training a smaller student to mimic a larger pretrained teacher.
        """
        logger.info(f"Performing distillation from {self.cfg.distillation.full_cfg_path}")

        default_cfg = get_default_config()
        distillation_cfg = OmegaConf.load(self.cfg.distillation.full_cfg_path)
        distillation_cfg = OmegaConf.merge(default_cfg, distillation_cfg)

        assert distillation_cfg.ibot.separate_head is True
        assert (
            distillation_cfg.ibot.head_n_prototypes == self.cfg.ibot.head_n_prototypes
        ), f"{distillation_cfg.ibot.head_n_prototypes} != {self.cfg.ibot.head_n_prototypes}"
        assert (
            distillation_cfg.dino.head_n_prototypes == self.cfg.dino.head_n_prototypes
        ), f"{distillation_cfg.dino.head_n_prototypes} != {self.cfg.dino.head_n_prototypes}"
        assert distillation_cfg.student.patch_size == self.cfg.student.patch_size

        teacher_model_dict = dict()

        backbone, embed_dim = build_model_from_cfg(distillation_cfg, only_teacher=True)
        teacher_model_dict["backbone"] = backbone

        teacher_model_dict["dino_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.dino.head_n_prototypes,
            hidden_dim=distillation_cfg.dino.head_hidden_dim,
            bottleneck_dim=distillation_cfg.dino.head_bottleneck_dim,
            nlayers=distillation_cfg.dino.head_nlayers,
        )
        teacher_model_dict["ibot_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.ibot.head_n_prototypes,
            hidden_dim=distillation_cfg.ibot.head_hidden_dim,
            bottleneck_dim=distillation_cfg.ibot.head_bottleneck_dim,
            nlayers=distillation_cfg.ibot.head_nlayers,
        )
        self.teacher = nn.ModuleDict(teacher_model_dict)

    def init_weights(self) -> None:
        """
        Initialize all model weights.

        This method handles the complete weight initialization pipeline:

        1. **Student initialization**: Backbone, DINO head, and iBOT head
        2. **Loss module initialization**: DINO and iBOT centering buffers
        3. **Teacher synchronization**: Copy student weights to EMA teacher
        4. **Gram teacher loading**: Load from checkpoint if configured
        5. **Resume from checkpoint**: Optionally load pretrained student weights
        6. **Distillation teacher loading**: Load teacher for knowledge distillation

        Must be called after :meth:`__init__` and before training.

        Note:
        ----
        Weights are initially set to NaN in ``build_model_from_cfg`` using
        ``torch.device("meta")``. This ensures all parameters are explicitly
        initialized and catches any uninitialized weights.

        Raises:
        ------
        ValueError
            If ``gram.use_loss=True`` but no checkpoint path is provided
            and ``gram.it_load_ema_teacher`` is not set.
        """
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        if self.has_gram_teacher:
            if self.gram_ckpt is not None:
                logger.info(f"Loading pretrained weights from {self.gram_ckpt}")
                init_fsdp_model_from_checkpoint(
                    self.gram_teacher,
                    self.gram_ckpt,
                    skip_load_keys=[
                        "dino_head",
                        "ibot_head",
                        "dino_loss.center",
                        "ibot_patch_loss.center",
                    ],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                    process_group=distributed.get_default_process_group(),
                )
                self.gram_teacher_initialized = True
            else:
                raise ValueError(f"Provide a correct path to {self.gram_ckpt}")
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
        if self.cfg.student.resume_from_teacher_chkpt:
            logger.info(f"Loading pretrained weights from {self.cfg.student.resume_from_teacher_chkpt}")
            init_fsdp_model_from_checkpoint(
                self.student,
                self.cfg.student.resume_from_teacher_chkpt,
                skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                process_group=distributed.get_process_subgroup(),
            )
            self.model_ema.load_state_dict(self.student.state_dict())
        if self.cfg.distillation.enabled:
            if self.cfg.distillation.checkpoint_path != "ignore":
                logger.info(f"Loading teacher to distil from : {self.cfg.distillation.checkpoint_path}")
                init_fsdp_model_from_checkpoint(
                    self.teacher,
                    self.cfg.distillation.checkpoint_path,
                    skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                    process_group=distributed.get_default_process_group(),
                )
            else:
                logger.info("Init teacher to distil from, used for testing purpose only")
                self.teacher.backbone.init_weights()
                self.teacher.dino_head.init_weights()
                self.teacher.ibot_head.init_weights()
            logger.info(f"Performing distillation from: {self.teacher}")

    def forward_backward(
        self,
        data: dict[str, Any],
        *,
        teacher_temp: float,
        iteration: int = 0,
        **ignored_kwargs: Any,
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        """
        Execute complete forward and backward pass for one training step.

        This is the main training method that:

        1. Moves data to GPU
        2. Computes teacher outputs (no gradients)
        3. Computes student outputs (with gradients)
        4. Computes all losses (DINO, iBOT, KoLeo, Gram)
        5. Executes backward pass

        Parameters:
        ----------
        data : dict[str, Any]
            Batch from the data loader containing:

            - ``collated_global_crops``: [2*B, 3, H, W] global crop images
            - ``collated_local_crops``: [n_local*B, 3, h, w] local crop images
            - ``collated_masks``: [2*B, P] boolean masks for iBOT
            - ``mask_indices_list``: [N] indices of masked patches
            - ``masks_weight``: [N] importance weights for masked patches
            - ``n_masked_patches``: [2*B] number of masked patches per image
            - ``collated_gram_teacher_crops``: (optional) crops for gram teacher
            - ``global_batch_size``: Total batch size across all ranks
            - ``upperbound``: Upper bound for various computations

        teacher_temp : float
            Temperature for teacher softmax. Lower = sharper distributions.
            Typically starts at 0.04 and may warm up to 0.07.

        iteration : int, default=0
            Current training iteration. Used for:
            - Loss weight scheduling (e.g., Gram loss warmup)
            - DINO local loss reweighting schedule

        **ignored_kwargs : Any
            Additional keyword arguments (ignored for compatibility).

        Returns:
        -------
        tuple[Tensor, dict[str, float | Tensor]]
            - ``loss_accumulator``: Scalar tensor with total weighted loss
            - ``metrics_dict``: Dictionary of individual losses and metrics for logging:
              - ``local_batch_size``: Per-GPU batch size
              - ``global_batch_size``: Total batch size
              - ``dino_local_crops_loss``: DINO loss on local crops
              - ``dino_global_crops_loss``: DINO loss on global crops
              - ``koleo_loss``: KoLeo uniformity loss
              - ``ibot_loss``: iBOT masked patch loss
              - ``gram_loss``: (if enabled) Gram matrix loss
              - Various loss weights and auxiliary statistics

        Note:
        ----
        This method calls ``loss.backward()`` internally. The optimizer
        step should be called after this method returns.

        Example:
        -------
        .. code-block:: python

            for data in dataloader:
                loss, metrics = model.forward_backward(
                    data,
                    teacher_temp=0.04,
                    iteration=step,
                )
                optimizer.step()
                optimizer.zero_grad()
                model.update_ema(m=ema_schedule[step])
        """
        del ignored_kwargs
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        if self.has_gram_teacher:
            assert (
                "collated_gram_teacher_crops" in data
            ), "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
            gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
        else:
            gram_teacher_crops = None

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        # Gram output
        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=global_crops.shape[-1],
            )
        else:
            gram_global = {}

        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )

        self.backprop_loss(loss_accumulator)

        # Return total weighted loss and a dict of metrics to log
        return loss_accumulator, metrics_dict | loss_dict

    @torch.no_grad()
    def get_teacher_output(
        self,
        images: Tensor,
        *,
        upperbound: int,
        mask_indices_list: Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute teacher outputs for global crops (no gradients).

        The teacher processes global crops and produces:
        - CLS tokens for DINO loss
        - Patch tokens for iBOT loss (only at masked positions)

        Outputs are centered using Sinkhorn-Knopp normalization to prevent
        collapse and ensure balanced prototype usage.

        Parameters:
        ----------
        images : Tensor
            Global crop images, shape [n_crops, B, 3, H, W].
            Typically n_crops=2 for two global views.

        upperbound : int
            Upper bound for computation (used internally).

        mask_indices_list : Tensor
            Flat indices of masked patches across all images.
            Shape [N] where N is total number of masked patches.

        teacher_temp : float
            Temperature for Sinkhorn-Knopp centering.

        n_masked_patches_tensor : Tensor
            Number of masked patches per image, shape [n_crops * B].

        Returns:
        -------
        dict[str, Tensor]
            Teacher outputs:

            - ``cls_pre_head``: [n_crops, B, D] CLS tokens before projection
            - ``reg_pre_head``: [n_crops, B, R, D] Register tokens
            - ``patch_pre_head``: [n_crops, B, P, D] All patch tokens
            - ``cls_after_head``: [n_crops, B, K] CLS after DINO head
            - ``cls_centered``: [n_crops, B, K] Sinkhorn-centered CLS
            - ``masked_patch_centered``: [N, K] Centered masked patch tokens
        """
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
            "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),  # [n_crops, B, P, D]
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    def get_gram_teacher_output(
        self,
        images: Tensor | None,
        *,
        masks: Tensor,
        teacher_global: dict[str, Tensor],
        student_global: dict[str, Tensor],
        student_global_crops_size: int,
    ) -> dict[str, Tensor]:
        """
        Compute Gram teacher outputs for Gram loss computation.

        The Gram loss compares feature correlation matrices (Gram matrices)
        between student and a reference teacher. The teacher can be:

        1. **EMA teacher** (``gram.ema_teacher=True``): Uses the same teacher
           as DINO/iBOT losses. No separate forward pass needed.

        2. **Separate teacher** (``gram.ema_teacher=False``): Uses a dedicated
           frozen backbone, possibly with different resolution or architecture.

        Parameters:
        ----------
        images : Tensor | None
            Gram teacher input crops, shape [n_crops, B, 3, H', W'].
            Can be different resolution than student crops.
            None if using EMA teacher.

        masks : Tensor
            Boolean mask indicating which patches are masked for iBOT.
            Shape [n_crops * B, P].

        teacher_global : dict[str, Tensor]
            Outputs from :meth:`get_teacher_output` (used if EMA teacher).

        student_global : dict[str, Tensor]
            Outputs from :meth:`get_student_output`.

        student_global_crops_size : int
            Spatial size of student global crops (e.g., 224).

        Returns:
        -------
        dict[str, Tensor]
            Gram computation inputs:

            - ``student_patches``: Student features for Gram loss
            - ``teacher_patches``: Teacher features for Gram loss
            - ``orig_student_patches``: All student patches (for stats)
            - ``orig_teacher_patches``: All teacher patches (for stats)

        Note:
        ----
        If teacher has different resolution, features are interpolated
        to match student resolution using the configured method
        (``gram.global_teacher_resize_method``).
        """
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]

        # Get gram targets
        if self.gram_ema_teacher:
            teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]
        else:
            if not self.gram_teacher_initialized:
                raise ValueError("Gram teacher has not been initialized. Load a checkpoint or from the EMA teacher.")
            n_crops, B, rgb, H, W = images.shape
            images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]

            with torch.no_grad():
                backbone_out = self.gram_teacher.backbone(images, is_training=True)
            teacher_patches = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P_T, D]

            # Downsample Gram teacher features if needed
            if teacher_patches.shape[1] != student_patches.shape[1]:
                N = H // self.cfg.student.patch_size
                assert teacher_patches.shape[1] == N**2
                N_student = student_global_crops_size // self.cfg.student.patch_size
                assert student_patches.shape[1] == N_student**2
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))  # [n_crops * B, D, N, N]
                patches_hw = torch.nn.functional.interpolate(
                    patches_hw,
                    size=(N_student, N_student),
                    mode=self.gram_global_teacher_resize_method,
                    align_corners=False,
                    antialias=self.gram_global_teacher_resize_antialias,
                )
                teacher_patches = patches_hw.flatten(-2, -1).transpose(
                    -2, -1
                )  # [n_crops * B, N_student * N_student, D]
                assert teacher_patches.shape == student_patches.shape

        # Select the patches to be considered in the loss
        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram_tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram_tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            "teacher_patches": teacher_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            # Unmasked patches, for computing statistics
            "orig_student_patches": orig_student_patches,  # [n_crops * B, P, D]
            "orig_teacher_patches": orig_teacher_patches,  # [n_crops * B, P, D]
        }

    def get_student_output(
        self,
        *,
        global_crops: Tensor,
        local_crops: Tensor,
        upperbound: int,
        masks: Tensor,
        mask_indices_list: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Compute student outputs for both global and local crops.

        The student processes:
        - **Global crops** with iBOT masking: Some patches are masked and
          the student must predict their teacher representations
        - **Local crops** without masking: Only DINO loss on CLS tokens

        Both crop types share the same backbone forward pass for efficiency.

        Parameters:
        ----------
        global_crops : Tensor
            Global view images, shape [n_global, B, 3, H, W].

        local_crops : Tensor
            Local view images, shape [n_local, B, 3, h, w].

        upperbound : int
            Upper bound for computation.

        masks : Tensor
            Boolean masks for iBOT, shape [n_global * B, P].
            True = masked (student must predict these).

        mask_indices_list : Tensor
            Flat indices of masked patches, shape [N].

        Returns:
        -------
        tuple[dict[str, Tensor], dict[str, Tensor]]
            - ``global_out``: Global crop outputs with keys:
              - ``cls_pre_head``: [n_global, B, D]
              - ``cls_after_head``: [n_global, B, K]
              - ``patch_pre_head``: [n_global, B, P, D]
              - ``masked_patch_after_head``: [N, K] (for iBOT)
              - ``masked_patch_pre_head``: [N, D]
            - ``local_out``: Local crop outputs with keys:
              - ``cls_pre_head``: [n_local, B, D]
              - ``cls_after_head``: [n_local, B, K]
              - ``patch_pre_head``: [n_local, B, P, D]
        """
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks if not self.is_distillation_enabled else None, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def compute_losses(
        self,
        *,
        teacher_global: dict[str, Tensor],
        student_global: dict[str, Tensor],
        student_local: dict[str, Tensor],
        gram_global: dict[str, Tensor],
        masks: Tensor,
        mask_indices_list: Tensor,
        masks_weight: Tensor,
        iteration: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute all loss terms and return weighted sum.

        Loss Components:
        ---------------
        1. **DINO Global Loss**: Student global CLS vs Teacher global CLS
        2. **DINO Local Loss**: Student local CLS vs Teacher global CLS
        3. **KoLeo Loss**: Uniformity regularization on student CLS features
        4. **iBOT Loss**: Masked patch prediction (student vs teacher)
        5. **Gram Loss** (optional): Feature correlation matching

        Loss Scaling:
        ------------
        DINO losses are scaled by the relative number of cross-entropy terms:
        - Global scale = n_global_terms / total_terms
        - Local scale = n_local_terms / total_terms

        This ensures equal per-term contribution regardless of crop counts.

        Parameters:
        ----------
        teacher_global : dict[str, Tensor]
            Teacher outputs from :meth:`get_teacher_output`.

        student_global : dict[str, Tensor]
            Student global crop outputs from :meth:`get_student_output`.

        student_local : dict[str, Tensor]
            Student local crop outputs from :meth:`get_student_output`.

        gram_global : dict[str, Tensor]
            Gram loss inputs from :meth:`get_gram_teacher_output`.

        masks : Tensor
            iBOT masks, shape [n_global * B, P].

        mask_indices_list : Tensor
            Indices of masked patches.

        masks_weight : Tensor
            Importance weights for masked patches.

        iteration : int
            Current iteration (for scheduled loss weights).

        Returns:
        -------
        tuple[Tensor, dict[str, Tensor]]
            - ``loss_accumulator``: Total weighted loss (scalar)
            - ``loss_dict``: Individual losses for logging
        """
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # Gram loss
        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            if self.gram_loss_schedule is not None:
                gram_loss_weight = self.gram_loss_schedule[iteration]
            else:
                gram_loss_weight = self.gram_loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    # Save stats over masked / unmasked tokens
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        return loss_accumulator, loss_dict

    @torch.no_grad()
    def gram_load_ema_teacher(self) -> None:
        """
        Load EMA teacher weights into the Gram teacher backbone.

        This is used when ``gram.it_load_ema_teacher`` is set, allowing
        the Gram teacher to be initialized from the EMA teacher after
        some training iterations rather than from a checkpoint.

        Only copies backbone weights; DINO and iBOT heads are skipped.
        """
        if self.has_gram_teacher:
            skip_load_prefixes = ["dino_head.", "ibot_head."]
            self.gram_teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.model_ema.state_dict().items()
                    if not any(k.startswith(prefix) for prefix in skip_load_prefixes)
                }
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
            self.gram_teacher_initialized = True

    def train(self) -> None:
        """
        Set model to training mode.

        Overrides ``nn.Module.train()`` to ensure teacher networks
        remain in eval mode (frozen, no dropout/batchnorm updates).
        """
        super().train()
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()

    def forward(self, inputs: Any) -> None:
        """
        Standard forward pass (not implemented).

        Use :meth:`forward_backward` instead for training.
        This method exists only for nn.Module compatibility.

        Raises:
        ------
        NotImplementedError
            Always raised. Use ``forward_backward`` for training.
        """
        raise NotImplementedError

    def backprop_loss(self, loss: Tensor) -> None:
        """
        Execute backward pass on the loss.

        Parameters:
        ----------
        loss : Tensor
            Scalar loss tensor to backpropagate.

        Note:
        ----
        This is a simple wrapper that can be overridden for custom
        gradient handling (e.g., gradient scaling, clipping).
        """
        loss.backward()

    def update_ema(self, m: float) -> None:
        """
        Update teacher weights via Exponential Moving Average.

        The update rule is:
        ``teacher = m * teacher + (1 - m) * student``

        Parameters:
        ----------
        m : float
            EMA momentum/decay factor in [0, 1].
            Higher values = slower teacher updates.

            - ``m=0.0``: Teacher becomes exact copy of student
            - ``m=0.99``: Teacher updated with 1% of student weights
            - ``m=0.999``: Teacher updated with 0.1% of student weights

        Note:
        ----
        Uses ``torch._foreach_*`` operations for efficient batched
        parameter updates. Parameter lists are cached after first call.

        Example:
        -------
        Typical momentum schedule starts at 0.996 and increases to 0.9999:

        .. code-block:: python

            # Linear warmup of momentum
            ema_momentum = 0.996 + (0.9999 - 0.996) * min(1, step / warmup_steps)
            model.update_ema(ema_momentum)
        """
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m: float = 0) -> None:
        """
        Update Gram teacher weights from the main teacher.

        Used when ``gram.rep_update=True`` to periodically refresh
        the Gram teacher backbone from the EMA teacher.

        Parameters:
        ----------
        m : float, default=0
            EMA momentum (0 = complete replacement with teacher weights).

        Note:
        ----
        Only called if ``gram.rep_update=True`` and iteration >
        ``gram.it_first_update`` at frequency ``gram.update_frequency``.
        """
        if not self.has_gram_teacher:
            return
        logger.info("Updating gram teacher with teacher weights.")
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(self.gram_teacher[k].parameters(), self.teacher[k].parameters()):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)

    def build_data_augmentation_dino(self, cfg: DictConfig) -> DataAugmentationDINO:
        """
        Build the DINO data augmentation pipeline.

        Creates a ``DataAugmentationDINO`` instance configured for multi-crop
        training with global and local views, color jittering, and optional
        Gram teacher crops.

        Parameters:
        ----------
        cfg : DictConfig
            Configuration object with ``crops`` section.

        Returns:
        -------
        DataAugmentationDINO
            Configured augmentation pipeline for training.
        """
        return DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
        )

    def get_maybe_fused_params_for_submodel(self, m: nn.Module) -> list[dict[str, Any]]:
        """
        Get parameter groups for a submodel with optional fusion.

        Creates parameter groups with:
        - Layer-wise learning rate decay
        - Special learning rate for patch embedding
        - Custom weight decay for DINO head

        Optionally fuses parameter groups for more efficient optimizer
        operations (``cfg.optim.multi_tensor_optim=True``).

        Parameters:
        ----------
        m : nn.Module
            Submodel (e.g., backbone, dino_head, ibot_head).

        Returns:
        -------
        list[dict[str, Any]]
            List of parameter group dicts for optimizer construction.
        """
        params_groups = get_params_groups_with_decay_fsdp(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
        )
        if self.cfg.optim.multi_tensor_optim:
            fused_params_groups = fuse_params_groups(params_groups)
            logger.info("fusing param groups")

            for g in fused_params_groups:
                g["foreach"] = True
                g["fused"] = True
            return fused_params_groups
        else:
            return params_groups

    def get_params_groups(self) -> list[dict[str, Any]]:
        """
        Get all parameter groups for optimizer construction.

        Collects parameter groups from all student submodels (backbone,
        dino_head, ibot_head) with appropriate learning rate and weight
        decay settings.

        Returns:
        -------
        list[dict[str, Any]]
            Combined parameter groups for all student components.
            Ready to pass to optimizer constructor.

        Example:
        -------
        .. code-block:: python

            param_groups = model.get_params_groups()
            optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
        """
        all_params_groups = []
        for name, m in self.student.items():
            logger.info(f"Getting paramer groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self) -> None:
        """
        Apply FSDP wrapping and compilation for distributed training.

        This method:

        1. Wraps student model with FSDP for gradient sharding
        2. Wraps teacher models for inference-only FSDP
        3. Optionally applies ``torch.compile`` for optimization
        4. Applies activation checkpointing if configured

        Must be called after :meth:`init_weights` and before training.

        Note:
        ----
        Uses different process groups for different models:

        - Student: Uses process subgroup (for gradient accumulation)
        - EMA Teacher: Uses process subgroup (mirrors student)
        - Gram Teacher: Uses default process group
        - Distillation Teacher: Uses default process group
        """
        process_subgroup = distributed.get_process_subgroup()
        default_process_group = distributed.get_default_process_group()
        inference_only_models = [self.model_ema]
        inference_only_models_process_groups = [process_subgroup]
        if self.has_gram_teacher:
            inference_only_models.append(self.gram_teacher)
            inference_only_models_process_groups.append(default_process_group)
        if self.cfg.distillation.enabled:
            inference_only_models.append(self.teacher)
            inference_only_models_process_groups.append(default_process_group)
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=inference_only_models,
            cfg=self.cfg,
            trained_model_process_group=process_subgroup,
            inference_only_models_process_groups=inference_only_models_process_groups,
        )

    def broadcast_to_subgroups(
        self,
        tensor: Tensor,
        over_dim: int,
        global_batch_size: int | None = None,
    ) -> Tensor:
        """
        Gather tensor globally then scatter to process subgroups.

        This operation enables communication patterns where data from all
        ranks needs to be redistributed to smaller process subgroups
        (e.g., for gradient accumulation across subsets of GPUs).

        Parameters:
        ----------
        tensor : Tensor
            Input tensor to broadcast.

        over_dim : int
            Dimension along which to concatenate gathered tensors.

        global_batch_size : int | None, optional
            If provided, truncate gathered tensor to this size along
            ``over_dim`` (handles uneven batch distribution).

        Returns:
        -------
        Tensor
            Redistributed tensor for the current process subgroup.
        """
        world_size = distributed.get_world_size()
        subgroup_size = distributed.get_subgroup_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered, tensor)
        catted = torch.cat(gathered, dim=over_dim)
        if global_batch_size is not None:
            catted = catted.narrow(dim=over_dim, start=0, length=global_batch_size)

        return catted.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()
