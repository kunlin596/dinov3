# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Multi-Student Knowledge Distillation Meta-Architecture.

This module implements the ``MultiDistillationMetaArch`` class for training multiple
student models simultaneously from a shared teacher. Each student runs on a subset
of GPUs (process subgroup) while sharing teacher outputs via broadcast.

Architecture Overview:
---------------------
::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      Multi-Distillation Training                        │
    │                                                                         │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    Shared Teacher (Frozen)                      │   │
    │   │                                                                 │   │
    │   │   Global Crops ──► Teacher Backbone ──► DINO/iBOT Heads         │   │
    │   │        │                                      │                 │   │
    │   │        │              ┌───────────────────────┘                 │   │
    │   │        │              ▼                                         │   │
    │   │        │     broadcast_to_subgroups()                           │   │
    │   │        │              │                                         │   │
    │   └────────┼──────────────┼─────────────────────────────────────────┘   │
    │            │              │                                             │
    │   ┌────────┼──────────────┼─────────────────────────────────────────┐   │
    │   │        ▼              ▼                                         │   │
    │   │   ┌─────────┐   ┌─────────┐   ┌─────────┐                       │   │
    │   │   │Student A│   │Student B│   │Student C│  (different archs)    │   │
    │   │   │(ranks   │   │(ranks   │   │(ranks   │                       │   │
    │   │   │ 0-3)    │   │ 4-5)    │   │ 6-7)    │                       │   │
    │   │   └────┬────┘   └────┬────┘   └────┬────┘                       │   │
    │   │        │             │             │                            │   │
    │   │        ▼             ▼             ▼                            │   │
    │   │   ┌─────────────────────────────────────┐                       │   │
    │   │   │     DINO + iBOT + KoLeo Losses      │                       │   │
    │   │   │   (computed per student subgroup)   │                       │   │
    │   │   └─────────────────────────────────────┘                       │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

Key Differences from Standard SSLMetaArch:
-----------------------------------------
1. **Subgroup Broadcasting**: Teacher outputs are broadcast to student subgroups
   rather than computed per-rank.

2. **Resolution Scaling**: Teacher crops can be downsampled to match student
   resolution (for training smaller students from larger teacher features).

3. **Simplified Configuration**: Fixed settings for losses (DINO, iBOT, KoLeo
   always computed), centering (Sinkhorn-Knopp), and crop types (global + local).

4. **No Gram Loss**: Multi-distillation focuses on knowledge transfer, not
   late-stage feature anchoring.

Usage Example:
-------------
Multi-distillation is typically configured via YAML:

.. code-block:: yaml

    # multidist_config.yaml
    multidistillation:
      enabled: true
      global_batch_size: 512
      students:
        - name: vits_student
          config_path: configs/train/vits_student.yaml
          ranks_range: [0, 4]  # GPUs 0-3
        - name: vitb_student
          config_path: configs/train/vitb_student.yaml
          ranks_range: [4, 8]  # GPUs 4-7

Then launched with:

.. code-block:: bash

    python scripts/dinov3_cli.py distill --config-file multidist_config.yaml

See Also:
--------
- :class:`SSLMetaArch`: Base class with full DINO/iBOT/Gram implementation
- :func:`setup_multidistillation`: Configuration setup for multi-distillation
- ``04_knowledge_distillation.md``: Tutorial on distillation workflows
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from .ssl_meta_arch import SSLMetaArch

logger = logging.getLogger("dinov3")


class MultiDistillationMetaArch(SSLMetaArch):
    """
    Multi-student knowledge distillation meta-architecture.

    Extends :class:`SSLMetaArch` to support training multiple student models
    simultaneously from a shared teacher. Each student operates on a process
    subgroup while teacher outputs are broadcast across subgroups.

    This architecture enables efficient knowledge transfer to multiple students
    of different sizes (e.g., ViT-S, ViT-B, ViT-L) in a single training run,
    with the teacher providing consistent soft targets to all students.

    Key Simplifications vs SSLMetaArch:
    ----------------------------------
    - **Fixed loss scales**: DINO, KoLeo, and iBOT losses use baked-in weights
    - **Always global + local crops**: No option to disable either crop type
    - **Separate heads**: Always uses separate DINO and iBOT heads
    - **Sinkhorn-Knopp centering**: Always used for teacher soft targets
    - **Per-GPU KoLeo**: Non-distributed computation for efficiency
    - **No Gram loss**: Gram anchoring is not used in distillation

    Attributes:
    ----------
    Inherits all attributes from :class:`SSLMetaArch`.

    Note:
    ----
    The teacher model processes crops at full resolution, then broadcasts
    features to student subgroups. Students may operate at lower resolution
    (controlled by ``crops.teacher_to_student_resolution_scale``).
    """

    def forward_backward(
        self,
        data: dict[str, Tensor],
        *,
        teacher_temp: float,
        iteration: int = 0,
        **ignored_kwargs,
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        """
        Execute forward pass, loss computation, and backward pass.

        This method orchestrates the multi-distillation training step:

        1. Extract crops and masks from data batch
        2. Optionally downsample crops for student resolution
        3. Broadcast teacher outputs to student subgroups
        4. Compute student outputs on subgroup-local data
        5. Compute DINO, iBOT, and KoLeo losses
        6. Backpropagate gradients

        Parameters:
        ----------
        data : dict[str, Tensor]
            Batch dictionary containing:

            - ``collated_global_crops``: [2*B, C, H, W] global crop images
            - ``collated_local_crops``: [n_local*B, C, h, w] local crop images
            - ``collated_masks``: [2*B, P] boolean masks for iBOT
            - ``mask_indices_list``: Indices of masked patches
            - ``masks_weight``: Per-mask loss weights
            - ``n_masked_patches``: Count of masked patches per image
            - ``global_batch_size``: Total batch size across all ranks
            - ``upperbound``: Upper bound for masked patch indices

        teacher_temp : float
            Temperature for teacher softmax (controls sharpness of soft targets).

        iteration : int, default=0
            Current training iteration (used for scheduling).

        **ignored_kwargs
            Additional kwargs are ignored (for API compatibility).

        Returns:
        -------
        tuple[Tensor, dict[str, float | Tensor]]
            - Total weighted loss scalar for logging
            - Dictionary of loss components and metrics:
              - ``batch_size``: Local batch size
              - ``dino_local_crops_loss``: DINO loss on local crops
              - ``dino_global_crops_loss``: DINO loss on global crops
              - ``ibot_loss``: iBOT masked patch prediction loss
              - ``koleo_loss``: KoLeo uniformity loss
        """
        del ignored_kwargs
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        B_teacher = B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["batch_size"] = B

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)
        global_batch_size = data["global_batch_size"]

        # Multidistillation codepath:

        # Downsample teacher crops to match student resolution
        downsampling_factor = getattr(self, "crops.teacher_to_student_resolution_scale", 1.0)
        if downsampling_factor != 1.0:
            global_crops = torch.nn.functional.interpolate(
                global_crops,
                scale_factor=1.0 / downsampling_factor,
                mode="bilinear",
                antialias=True,
            )
        global_crops_subgroup = self.broadcast_to_subgroups(
            global_crops.view(n_global_crops, -1, *global_crops.shape[1:]),
            1,
            global_batch_size=global_batch_size,
        ).view(-1, *global_crops.shape[1:])
        local_crops_subgroup = self.broadcast_to_subgroups(
            local_crops.view(n_local_crops, -1, *local_crops.shape[1:]),
            1,
            global_batch_size=global_batch_size,
        ).view(-1, *local_crops.shape[1:])
        B = local_crops_subgroup.shape[0] // n_local_crops

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B_teacher)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
            global_batch_size=global_batch_size,
        )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = self.get_student_output(
            global_crops=global_crops_subgroup.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops_subgroup.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )
        # End of multidistillation codepath

        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            gram_global=None,
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
        global_batch_size: int,
    ) -> dict[str, Tensor]:
        """
        Compute teacher outputs and broadcast to student subgroups.

        This method extends the base class to support multi-distillation by:

        1. Computing teacher backbone features on full-resolution crops
        2. Broadcasting intermediate features to all student subgroups
        3. Completing head computations after broadcast (for efficiency)
        4. Applying Sinkhorn-Knopp centering for soft targets

        The two-stage head computation (before and after broadcast) reduces
        communication overhead by broadcasting lower-dimensional intermediate
        representations instead of final outputs.

        Parameters:
        ----------
        images : Tensor
            Global crop images, shape [n_crops, B_teacher, C, H, W].

        upperbound : int
            Upper bound for valid indices in mask_indices_list.

        mask_indices_list : Tensor
            Flattened indices of masked patches for iBOT loss.

        teacher_temp : float
            Temperature for Sinkhorn-Knopp soft target normalization.

        n_masked_patches_tensor : Tensor
            Number of masked patches per image (for weighted averaging).

        global_batch_size : int
            Total batch size across all ranks (for broadcast sizing).

        Returns:
        -------
        dict[str, Tensor]
            Teacher output dictionary containing:

            - ``cls_after_head``: [n_crops, B, K] CLS token logits
            - ``cls_centered``: [n_crops, B, K] Sinkhorn-Knopp centered CLS
            - ``masked_patch_centered``: [n_masked, K] Centered masked patch logits

        Note:
        ----
        This method is decorated with ``@torch.no_grad()`` since teacher
        parameters are frozen and we only need forward computation.
        """
        n_crops, B_teacher, rgb, H, W = images.shape

        backbone_out = self.teacher.backbone(images.flatten(0, 1), is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        R, D = reg.shape[-2:]

        # Multidistillation codepath:
        # IBOT head only on patches that are masked for the student
        n_tokens = ibot_patch.shape[1]
        masked_patch_after_head = self.teacher.ibot_head(ibot_patch.flatten(0, 1), no_last_layer=True)
        masked_patch_after_head = masked_patch_after_head.view(n_crops, -1, *masked_patch_after_head.shape[1:])
        masked_patch_after_head = self.broadcast_to_subgroups(
            masked_patch_after_head,
            over_dim=1,
            global_batch_size=global_batch_size * n_tokens,
        )
        buffer = torch.index_select(masked_patch_after_head.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer, only_last_layer=True)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls, no_last_layer=True)  # [n_crops * B, K]
        cls_after_head = cls_after_head.view(n_crops, -1, *cls_after_head.shape[1:])
        cls_after_head = self.broadcast_to_subgroups(cls_after_head, over_dim=1, global_batch_size=global_batch_size)
        B = cls_after_head.shape[1]
        cls_after_head = cls_after_head.flatten(0, 1)
        cls_after_head = self.teacher.dino_head(cls_after_head, only_last_layer=True)  # [n_crops * B, K]
        # End of multidistillation codepath

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
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }
