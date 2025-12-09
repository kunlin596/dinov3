# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Data Augmentation Pipeline for DINOv3 Self-Supervised Learning.

This module implements the multi-crop data augmentation strategy that is central
to DINO-style self-supervised learning. The augmentation creates multiple views
of each image at different scales and with different transformations.

Multi-Crop Strategy:
-------------------
::

    Input Image
         │
         ├────────────────────────────────────────┐
         │                                        │
         ▼                                        ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │   Global Crops (2)  │              │   Local Crops (N)   │
    │                     │              │                     │
    │   • Large scale     │              │   • Small scale     │
    │     (0.32 - 1.0)    │              │     (0.05 - 0.32)   │
    │   • Size: 224x224   │              │   • Size: 96x96     │
    │   • Full context    │              │   • Local details   │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               ▼                                    ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │  Color Distortions  │              │  Color Distortions  │
    │  + GaussianBlur     │              │  + GaussianBlur     │
    │  + Solarization     │              │                     │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               ▼                                    ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │  Normalize to       │              │  Normalize to       │
    │  ImageNet stats     │              │  ImageNet stats     │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               └──────────────┬─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Output Dict    │
                    │                 │
                    │  global_crops   │  → Teacher & Student
                    │  local_crops    │  → Student only
                    │  gram_crops     │  → Gram Teacher (opt)
                    └─────────────────┘

Augmentation Components:
-----------------------
1. **Geometric Augmentations**
   - RandomResizedCrop: Scale-aware cropping
   - RandomHorizontalFlip: 50% probability flip

2. **Color Distortions**
   - ColorJitter: Brightness, contrast, saturation, hue
   - RandomGrayscale: 20% probability
   - Can be shared across all crops (``share_color_jitter=True``)

3. **Additional Augmentations**
   - GaussianBlur: Different probabilities per crop type
   - RandomSolarize: Only for global crop 2 (20% probability)

4. **Normalization**
   - ImageNet statistics by default
   - Converts to float32 tensor

Usage in Training:
-----------------
- **Global crops** → Both teacher and student process these
- **Local crops** → Only student processes these
- **DINO loss**: Student local/global vs Teacher global (cross-entropy)
- **iBOT loss**: Masked patches in global crops

Example:
-------
.. code-block:: python

    from dinov3.data.augmentations import DataAugmentationDINO

    # Create augmentation pipeline
    augment = DataAugmentationDINO(
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
    )

    # Apply to a PIL image
    output = augment(pil_image)

    # Access crops
    global_crop_1, global_crop_2 = output["global_crops"]  # [3, 224, 224]
    local_crops = output["local_crops"]  # List of [3, 96, 96]

See Also:
--------
- ``dinov3/data/collate.py``: Batching and iBOT mask generation
- ``dinov3/data/masking.py``: Mask sampling strategies
- ``dinov3/train/ssl_meta_arch.py``: How augmented data is consumed
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

if TYPE_CHECKING:
    from PIL import Image
    from numpy.typing import NDArray

logger = logging.getLogger("dinov3")


# ==============================================================================
# DINO Data Augmentation
# ==============================================================================


# ==============================================================================


class DataAugmentationDINO:
    """
    Multi-crop data augmentation pipeline for DINOv3 training.

    This class implements the augmentation strategy from DINO/DINOv2 that creates
    multiple views of each input image. The key insight is that comparing features
    from crops at different scales encourages the model to learn multi-scale,
    semantically meaningful representations.

    Crop Types:
    ----------
    1. **Global Crops** (2 crops)
       - Large-scale views covering most of the image
       - Scale range: typically (0.32, 1.0) of image area
       - Size: typically 224x224 pixels
       - Processed by both teacher and student networks
       - Different augmentation variants:
         - Global crop 1: Always GaussianBlur
         - Global crop 2: 10% GaussianBlur + 20% Solarize

    2. **Local Crops** (N crops, typically 8)
       - Small-scale views showing local image regions
       - Scale range: typically (0.05, 0.32) of image area
       - Size: typically 96x96 pixels
       - Processed by student network only
       - Encourages learning from local details

    3. **Gram Teacher Crops** (optional, 2 crops)
       - Separate crops for Gram loss computation
       - Can have different resolution than global crops
       - Optionally without color distortions

    Attributes:
    ----------
    global_crops_scale : tuple[float, float]
        Scale range for global crop random resize.

    local_crops_scale : tuple[float, float]
        Scale range for local crop random resize.

    local_crops_number : int
        Number of local crops per image.

    global_crops_size : int
        Output size of global crops in pixels.

    local_crops_size : int
        Output size of local crops in pixels.

    gram_teacher_crops_size : int | None
        Output size of Gram teacher crops, or None if disabled.

    share_color_jitter : bool
        If True, apply the same color jitter to all crops from an image.
        This ensures color consistency across views.

    Example:
    -------
    .. code-block:: python

        augment = DataAugmentationDINO(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=224,
            local_crops_size=96,
            share_color_jitter=False,  # Independent jitter per crop
        )

        # Input: PIL Image
        # Output: dict with crop tensors
        crops = augment(pil_image)

        # crops["global_crops"]: List[Tensor], 2 x [3, 224, 224]
        # crops["local_crops"]: List[Tensor], 8 x [3, 96, 96]

    Note:
    ----
    The augmentation pipeline is not deterministic by default. For reproducible
    training, set random seeds for numpy, torch, and torchvision.
    """

    def __init__(
        self,
        global_crops_scale: tuple[float, float],
        local_crops_scale: tuple[float, float],
        local_crops_number: int,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
        gram_teacher_crops_size: int | None = None,
        gram_teacher_no_distortions: bool = False,
        teacher_no_color_jitter: bool = False,
        local_crops_subset_of_global_crops: bool = False,
        patch_size: int = 16,
        share_color_jitter: bool = False,
        horizontal_flips: bool = True,
        mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
        std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    ) -> None:
        """
        Initialize the DINO data augmentation pipeline.

        Parameters:
        ----------
        global_crops_scale : tuple[float, float]
            Min and max scale for global crop RandomResizedCrop.
            Typical: (0.32, 1.0) meaning 32-100% of image area.

        local_crops_scale : tuple[float, float]
            Min and max scale for local crop RandomResizedCrop.
            Typical: (0.05, 0.32) meaning 5-32% of image area.

        local_crops_number : int
            Number of local crops to generate per image.
            Typical: 8-10 local crops.

        global_crops_size : int, default=224
            Output spatial size of global crops (square).

        local_crops_size : int, default=96
            Output spatial size of local crops (square).

        gram_teacher_crops_size : int | None, default=None
            Output size for Gram teacher crops. If None, Gram crops
            are not generated. Can differ from global_crops_size.

        gram_teacher_no_distortions : bool, default=False
            If True, Gram teacher crops skip color distortions.
            Useful for matching against undistorted reference features.

        teacher_no_color_jitter : bool, default=False
            If True, provide un-jittered crops for teacher network.
            Rarely used; most setups apply same augmentation to teacher.

        local_crops_subset_of_global_crops : bool, default=False
            If True, local crops are extracted from global crops
            rather than independently from the original image.
            Ensures local crops are spatially contained within globals.

        patch_size : int, default=16
            ViT patch size. Used only when ``local_crops_subset_of_global_crops=True``
            to ensure crop offsets align with patch boundaries.

        share_color_jitter : bool, default=False
            If True, apply identical color jitter to all crops.
            Creates more consistent multi-view appearance.

        horizontal_flips : bool, default=True
            Whether to apply random horizontal flips.
            Set to False for datasets with directional semantics.

        mean : tuple[float, float, float], default=IMAGENET_DEFAULT_MEAN
            Per-channel mean for normalization.

        std : tuple[float, float, float], default=IMAGENET_DEFAULT_STD
            Per-channel standard deviation for normalization.
        """
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        # ======================================================================
        # Log configuration
        # ======================================================================
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # ======================================================================
        # Geometric Augmentations (crop + flip)
        # ======================================================================

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # ======================================================================
        # Color Distortions
        # ======================================================================

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        # Global crop 1: Always apply GaussianBlur (asymmetric augmentation)
        global_transfo1_extra = GaussianBlur(p=1.0)

        # Global crop 2: Lighter blur (10%) + Solarization (20%)
        # This asymmetry between global crops is important for learning
        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        # Local crops: Moderate blur (50%), no solarization
        local_transfo_extra = GaussianBlur(p=0.5)

        # ======================================================================
        # ======================================================================
        # Normalization
        # ======================================================================
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        # ======================================================================
        # Compose Final Transforms
        # ======================================================================
        # Two modes: shared or independent color jitter
        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose([resize_global, color_jittering, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, color_jittering, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image: Image.Image) -> dict[str, Any]:
        """
        Apply augmentations to generate multi-crop views.

        Parameters:
        ----------
        image : PIL.Image.Image
            Input image in PIL format (RGB).

        Returns:
        -------
        dict[str, Any]
            Dictionary containing augmented crops:

            - ``global_crops``: List of 2 tensors, each [3, H, W]
              where H=W=global_crops_size. These are the main views
              for DINO loss computation.

            - ``global_crops_teacher``: List of 2 tensors, same shape.
              Usually identical to global_crops unless
              ``teacher_no_color_jitter=True``.

            - ``local_crops``: List of N tensors, each [3, h, w]
              where h=w=local_crops_size. Only student processes these.

            - ``gram_teacher_crops`` (if configured): List of 2 tensors
              for Gram loss computation.

            - ``offsets``: Tuple of (x, y) offsets if
              ``local_crops_subset_of_global_crops=True``, else empty tuple.

            - ``weak_flag``: Legacy flag (always True).

        Example:
        -------
        .. code-block:: python

            from PIL import Image

            img = Image.open("image.jpg").convert("RGB")
            output = augment(img)

            # Stack global crops for batch processing
            global_batch = torch.stack(output["global_crops"])  # [2, 3, 224, 224]
            local_batch = torch.stack(output["local_crops"])    # [8, 3, 96, 96]
        """
        output: dict[str, Any] = {}
        output["weak_flag"] = True  # Legacy flag from MUGS implementationmplementation

        # Apply shared color jitter first if enabled
        if self.share_color_jitter:
            image = self.color_jittering(image)

        # ======================================================================
        # Generate Global Crops
        # ======================================================================
        # Global crop 1: geometric aug -> transforms (blur) -> normalize -> resize
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        # Global crop 2: same pipeline but different blur/solarize
        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # ======================================================================
        # Generate Teacher Crops (may differ from student crops)
        # ======================================================================
        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # ======================================================================
        # Generate Gram Teacher Crops (optional)
        # ======================================================================
        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # ======================================================================
        # Generate Local Crops
        # ======================================================================
        # local crops:
        if self.local_crops_subset_of_global_crops:
            # Mode 1: Extract local crops as subregions of global crops
            # This ensures local views are spatially contained within global views
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            # Extract crops at patch-aligned offsets for ViT compatibility
            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            # Mode 2: Independent local crops from original image
            # Standard approach - local crops are independent random views
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output
