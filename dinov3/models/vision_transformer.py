# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 Vision Transformer (ViT) Implementation.

This module provides the core Vision Transformer architecture used in DINOv3
for self-supervised pretraining. The implementation supports:

- **Rotary Position Embeddings (RoPE)**: Resolution-agnostic position encoding
- **Multi-scale inputs**: Process global and local crops of different sizes
- **Masked image modeling**: iBOT-style masked patch prediction
- **Storage tokens**: Additional learnable tokens (registers) for improved attention

Architecture Overview:
---------------------
::

    Input Image [B, 3, H, W]
           │
           ▼
    ┌─────────────────┐
    │   PatchEmbed    │  Convolution: [B, 3, H, W] → [B, h, w, D]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Token Assembly │  [CLS] + [Storage×R] + [Patches×P]
    │  + Mask Tokens  │  Shape: [B, 1+R+P, D]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   RoPE Embed    │  Rotary position encoding for patches
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Transformer    │  N × SelfAttentionBlock
    │    Blocks       │  (Attention + FFN + LayerScale)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Layer Norm    │  Optional: separate norms for CLS/patches
    └────────┬────────┘
             │
             ▼
    Output Dict:
    - x_norm_clstoken: [B, D]     CLS token features
    - x_storage_tokens: [B, R, D]  Register features
    - x_norm_patchtokens: [B, P, D] Patch features

Model Variants:
--------------
- ``vit_small``: 384-dim, 12 layers, 6 heads (~22M params)
- ``vit_base``: 768-dim, 12 layers, 12 heads (~86M params)
- ``vit_large``: 1024-dim, 24 layers, 16 heads (~307M params)
- ``vit_so400m``: 1152-dim, 27 layers, 18 heads (~400M params)
- ``vit_huge2``: 1280-dim, 32 layers, 20 heads (~632M params)
- ``vit_giant2``: 1536-dim, 40 layers, 24 heads (~1.1B params)
- ``vit_7b``: 4096-dim, 40 layers, 32 heads (~7B params)

Usage Example:
-------------
.. code-block:: python

    from dinov3.models.vision_transformer import vit_large

    # Create ViT-Large model
    model = vit_large(
        patch_size=16,
        drop_path_rate=0.3,
        layerscale_init=1e-5,
    )
    model.init_weights()

    # Forward pass (training mode returns dict)
    images = torch.randn(4, 3, 224, 224)
    output = model(images, is_training=True)
    cls_features = output["x_norm_clstoken"]  # [4, 1024]
    patch_features = output["x_norm_patchtokens"]  # [4, 196, 1024]

    # Forward pass (inference mode returns CLS only)
    features = model(images, is_training=False)  # [4, 1024]

See Also:
--------
- :class:`SelfAttentionBlock`: Transformer block implementation
- :class:`RopePositionEmbedding`: Rotary position encoding
- :class:`PatchEmbed`: Image to patch embedding
- ``01_model_architecture.md``: Detailed architecture tutorial
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

# =============================================================================
# Layer Configuration Dictionaries
# =============================================================================

ffn_layer_dict: dict[str, type[nn.Module] | partial[nn.Module]] = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}
"""
FFN layer type mapping.

- ``mlp``: Standard MLP (Linear → GELU → Linear)
- ``swiglu``: SwiGLU activation (more expressive, used in LLaMA)
- ``swigluN``: SwiGLU with hidden dimension aligned to N (for hardware efficiency)
"""

norm_layer_dict: dict[str, type[nn.Module] | partial[nn.Module]] = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}
"""
Normalization layer type mapping.

- ``layernorm``: Standard LayerNorm with eps=1e-6
- ``layernormbf16``: LayerNorm with eps=1e-5 (better for bf16 training)
- ``rmsnorm``: Root Mean Square LayerNorm (faster, used in LLaMA)
"""

dtype_dict: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
"""PyTorch dtype mapping for RoPE embeddings."""


# =============================================================================
# Weight Initialization
# =============================================================================


def init_weights_vit(module: nn.Module, name: str = "") -> None:
    """
    Initialize Vision Transformer weights.

    Applies ViT-specific initialization to various layer types:

    - **Linear layers**: Truncated normal (std=0.02) for weights, zeros for bias
    - **LayerNorm/RMSNorm**: Default initialization via reset_parameters()
    - **LayerScale**: Default initialization via reset_parameters()
    - **PatchEmbed**: Default initialization via reset_parameters()

    This function is designed to be used with :func:`named_apply` to
    recursively initialize all modules in a model.

    Parameters:
    ----------
    module : nn.Module
        Module to initialize.

    name : str, default=""
        Name of the module (unused, for compatibility with named_apply).

    Example:
    -------
    .. code-block:: python

        from dinov3.utils import named_apply
        from dinov3.models.vision_transformer import init_weights_vit

        model = DinoVisionTransformer(...)
        named_apply(init_weights_vit, model)
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)  # type: ignore[union-attr]
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)  # type: ignore[union-attr]
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    """DINOv3 Vision Transformer backbone for self-supervised learning.

    This Vision Transformer implementation is optimized for DINOv3 pretraining
    with support for:

    - **Rotary Position Embeddings (RoPE)**: Resolution-agnostic 2D position encoding
    - **Storage/Register Tokens**: Additional learnable tokens to prevent attention collapse
    - **Masked Image Modeling**: Token masking support for iBOT loss
    - **Multi-scale Processing**: Handle global and local crops efficiently

    Architecture:
        The model processes images through: PatchEmbed → Token Assembly →
        Transformer Blocks → LayerNorm → Output Features.

    Args:
        img_size: Default input image size (used only for patch count calculation).
            The model supports any resolution due to RoPE. Default: 224.
        patch_size: Size of image patches. Default: 16.
        in_chans: Number of input image channels. Default: 3.
        pos_embed_rope_base: RoPE base frequency for position encoding. Higher
            values give coarser position sensitivity. Default: 100.0.
        pos_embed_rope_min_period: Minimum period for RoPE frequencies.
            Default: None (determined by embed_dim).
        pos_embed_rope_max_period: Maximum period for RoPE frequencies.
            Default: None (determined by embed_dim).
        pos_embed_rope_normalize_coords: How to normalize 2D coordinates.
            One of "min", "max", or "separate". Default: "separate".
        pos_embed_rope_shift_coords: Amount to shift coordinates. Default: None.
        pos_embed_rope_jitter_coords: Coordinate jittering for augmentation. Default: None.
        pos_embed_rope_rescale_coords: Coordinate rescaling factor. Default: None.
        pos_embed_rope_dtype: Data type for RoPE computation ("bf16" or "fp32").
            Default: "bf16".
        embed_dim: Transformer embedding dimension. Default: 768.
        depth: Number of transformer blocks. Default: 12.
        num_heads: Number of attention heads. Default: 12.
        ffn_ratio: Feed-forward network expansion ratio. Default: 4.0.
        qkv_bias: Whether to include bias in QKV projections. Default: True.
        drop_path_rate: Stochastic depth rate for all blocks. Default: 0.0.
        layerscale_init: Initial value for LayerScale (None to disable). Default: None.
        norm_layer: Normalization layer type ("layernorm" or "rmsnorm"). Default: "layernorm".
        ffn_layer: FFN layer type ("mlp", "swiglu", or "swiglu32"). Default: "mlp".
        ffn_bias: Whether to include bias in FFN layers. Default: True.
        proj_bias: Whether to include bias in attention output projection. Default: True.
        n_storage_tokens: Number of register/storage tokens (0 to disable). Default: 0.
        mask_k_bias: Whether to use masked key bias. Default: False.
        untie_cls_and_patch_norms: Use separate LayerNorm for CLS and patches. Default: False.
        untie_global_and_local_cls_norm: Use separate LayerNorm for global vs local
            CLS tokens during training. Default: False.
        device: Device to create parameters on. Default: None.
        **ignored_kwargs: Additional kwargs are logged and ignored.

    Attributes:
        embed_dim: Transformer embedding dimension.
        n_blocks: Number of transformer blocks.
        num_heads: Number of attention heads.
        patch_size: Image patch size.
        patch_embed: Patch embedding layer.
        cls_token: Learnable CLS token (shape: [1, 1, embed_dim]).
        storage_tokens: Register tokens (shape: [1, n_storage_tokens, embed_dim]).
        rope_embed: Rotary position embedding layer.
        blocks: ModuleList of SelfAttentionBlocks.
        norm: Output layer normalization.

    Example:
        >>> model = DinoVisionTransformer(
        ...     embed_dim=1024,
        ...     depth=24,
        ...     num_heads=16,
        ...     drop_path_rate=0.3,
        ... )
        >>> model.init_weights()
        >>> x = torch.randn(4, 3, 224, 224)
        >>> output = model(x, is_training=True)
        >>> output["x_norm_clstoken"].shape
        torch.Size([4, 1024])

    See Also:
        :func:`vit_large`: Factory function for ViT-Large configuration.
        :class:`SelfAttentionBlock`: Transformer block with RoPE support.
    """

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        """Initialize the DINOv3 Vision Transformer.

        This constructor builds all model components including patch embedding,
        positional encoding, transformer blocks, and normalization layers.

        **Important**: Call :meth:`init_weights` after construction to properly
        initialize all parameters.

        Args:
            img_size: Reference image size for calculating number of patches.
                Only used by PatchEmbed for initialization; the model accepts
                any resolution at runtime due to RoPE. Default: 224.

            patch_size: Size of each square image patch in pixels. The image is
                divided into non-overlapping patches of size ``(patch_size, patch_size)``.
                Common values: 14, 16. Smaller patches = more tokens = higher
                compute but finer spatial resolution. Default: 16.

            in_chans: Number of input image channels. Default: 3 (RGB).

            pos_embed_rope_base: Base frequency for Rotary Position Embeddings.
                Controls the wavelength of position encoding frequencies.
                Higher values → coarser position sensitivity (better for larger
                images). Lower values → finer position sensitivity. Default: 100.0.

            pos_embed_rope_min_period: Minimum period (wavelength) for RoPE
                frequency bands. Controls the highest frequency component.
                If None, determined automatically from embed_dim. Default: None.

            pos_embed_rope_max_period: Maximum period (wavelength) for RoPE
                frequency bands. Controls the lowest frequency component.
                If None, determined automatically from embed_dim. Default: None.

            pos_embed_rope_normalize_coords: How to normalize 2D patch coordinates
                before applying RoPE. Options:

                - ``"separate"``: Normalize H and W independently to [0, 1]
                - ``"min"``: Normalize by min(H, W) (preserves aspect ratio)
                - ``"max"``: Normalize by max(H, W) (preserves aspect ratio)

                Default: "separate".

            pos_embed_rope_shift_coords: Amount to shift normalized coordinates.
                Useful for centering the coordinate system. For example, 0.5
                shifts coordinates from [0, 1] to [-0.5, 0.5]. Default: None.

            pos_embed_rope_jitter_coords: Random jitter magnitude for coordinate
                augmentation during training. Adds uniform noise in
                [-jitter, +jitter] to coordinates. Default: None (no jitter).

            pos_embed_rope_rescale_coords: Multiplicative rescaling factor for
                coordinates. Applied after normalization. Default: None.

            pos_embed_rope_dtype: Data type for RoPE computation. Options:

                - ``"bf16"``: bfloat16 (recommended for training efficiency)
                - ``"fp16"``: float16
                - ``"fp32"``: float32 (highest precision)

                Default: "bf16".

            embed_dim: Dimension of token embeddings throughout the transformer.
                Also called ``d_model`` or hidden size. Must be divisible by
                ``num_heads``. Standard values by model size:

                - ViT-S: 384
                - ViT-B: 768
                - ViT-L: 1024
                - ViT-H: 1280
                - ViT-g: 1536

                Default: 768.

            depth: Number of transformer blocks (layers). More layers = more
                capacity but higher compute/memory. Standard values:

                - ViT-S/B: 12
                - ViT-L: 24
                - ViT-H: 32
                - ViT-g: 40

                Default: 12.

            num_heads: Number of attention heads. Each head has dimension
                ``embed_dim // num_heads``. More heads allow learning different
                attention patterns. Must evenly divide ``embed_dim``. Default: 12.

            ffn_ratio: Expansion ratio for the feed-forward network hidden
                dimension. FFN hidden dim = ``embed_dim * ffn_ratio``.
                Standard value is 4.0. SwiGLU variants may use ~2.67. Default: 4.0.

            qkv_bias: Whether to include bias terms in the Query, Key, Value
                projections of attention. Default: True.

            drop_path_rate: Stochastic depth (DropPath) probability. During
                training, randomly drops entire residual branches with this
                probability. Provides regularization for deep networks.
                Typical values: 0.0-0.5 depending on model size. Default: 0.0.

            layerscale_init: Initial value for LayerScale parameters. LayerScale
                multiplies each residual branch output by a learnable scalar,
                initialized to this value. Helps training stability for deep
                networks. Typical values: 1e-4 to 1e-6. If None, LayerScale
                is disabled. Default: None.

            norm_layer: Type of normalization layer. Options:

                - ``"layernorm"``: Standard LayerNorm with eps=1e-6
                - ``"layernormbf16"``: LayerNorm with eps=1e-5 (for bf16 stability)
                - ``"rmsnorm"``: RMSNorm (faster, no mean centering)

                Default: "layernorm".

            ffn_layer: Type of feed-forward network. Options:

                - ``"mlp"``: Standard MLP (Linear → GELU → Linear)
                - ``"swiglu"``: SwiGLU activation (Linear → SiLU ⊙ Linear → Linear)
                - ``"swiglu32/64/128"``: SwiGLU with hidden dim aligned to N

                SwiGLU is more expressive but has ~50% more parameters in the
                gating projection. Default: "mlp".

            ffn_bias: Whether to include bias terms in FFN linear layers.
                Default: True.

            proj_bias: Whether to include bias in the attention output
                projection (the final linear after combining heads). Default: True.

            n_storage_tokens: Number of additional learnable "register" or
                "storage" tokens prepended after CLS. These tokens:

                - Provide extra capacity for the model to store information
                - Help prevent attention collapse on background patches
                - Are positioned as: [CLS, Storage×R, Patches×P]

                Set to 0 to disable. Typical values: 0, 4, 8. Default: 0.

            mask_k_bias: Whether to use learnable bias specifically for masked
                key positions in attention. Used for iBOT masked image modeling.
                Default: False.

            untie_cls_and_patch_norms: If True, use separate LayerNorm for CLS
                token vs patch tokens in the output. This allows the model to
                learn different normalization statistics for global (CLS) and
                local (patch) features. Default: False.

            untie_global_and_local_cls_norm: If True, use a separate LayerNorm
                for local crop CLS tokens during training. This helps when
                global and local crops need different normalization. Only
                applies during training. Default: False.

            device: PyTorch device to create parameters on. If None, uses the
                default device. Default: None.

            **ignored_kwargs: Any additional keyword arguments are logged as
                warnings and ignored. This allows forward compatibility with
                config files that may have extra fields.

        Raises:
            KeyError: If ``norm_layer`` or ``ffn_layer`` is not a valid option.

        Example:
            >>> # Create ViT-Large with standard DINOv3 settings
            >>> model = DinoVisionTransformer(
            ...     embed_dim=1024,
            ...     depth=24,
            ...     num_heads=16,
            ...     drop_path_rate=0.4,
            ...     layerscale_init=1e-5,
            ...     n_storage_tokens=4,
            ...     ffn_layer="swiglu",
            ...     norm_layer="rmsnorm",
            ... )
            >>> model.init_weights()
            >>>
            >>> # Check model size
            >>> n_params = sum(p.numel() for p in model.parameters())
            >>> print(f"Parameters: {n_params / 1e6:.1f}M")
            Parameters: 307.0M

        Note:
            The model uses keyword-only arguments (``*``) to enforce explicit
            parameter naming and prevent positional argument errors.
        """
        super().__init__()

        # =================================================================
        # Step 1: Handle unknown kwargs for forward compatibility
        # =================================================================
        # Log warnings for any unrecognized parameters (allows loading configs
        # from newer versions without crashing)
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # =================================================================
        # Step 2: Resolve layer types from string names
        # =================================================================
        # Look up the normalization class from the registry
        # Options: "layernorm" → nn.LayerNorm, "rmsnorm" → RMSNorm
        norm_layer_cls = norm_layer_dict[norm_layer]

        # =================================================================
        # Step 3: Store core architecture hyperparameters as attributes
        # =================================================================
        # These are frequently accessed during forward pass and for model inspection
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        # =================================================================
        # Step 4: Create Patch Embedding layer
        # =================================================================
        # Converts image [B, C, H, W] → patch grid [B, H//patch_size, W//patch_size, embed_dim]
        # Uses a single convolution with kernel_size=stride=patch_size
        # flatten_embedding=False keeps spatial structure for RoPE position encoding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,  # Keep 2D structure: [B, H, W, D] not [B, H*W, D]
        )

        # =================================================================
        # Step 5: Create learnable special tokens
        # =================================================================
        # CLS token: Global representation token, prepended to sequence
        # Shape [1, 1, embed_dim] - will be broadcast to batch size during forward
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))

        # Storage/Register tokens: Additional learnable tokens that help with
        # attention pattern quality (prevent attention collapse on background)
        # Positioned after CLS: [CLS, Storage×R, Patches×P]
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        # =================================================================
        # Step 6: Create Rotary Position Embedding (RoPE)
        # =================================================================
        # RoPE encodes 2D positions directly into attention via rotation matrices
        # Key advantage: Resolution-agnostic (works with any image size at inference)
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )

        # =================================================================
        # Step 7: Create Transformer Blocks
        # =================================================================
        # Look up FFN class: "mlp" → Mlp, "swiglu" → SwiGLUFFN
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]

        # Create uniform FFN ratio for all blocks (could be varied per-block)
        ffn_ratio_sequence = [ffn_ratio] * depth

        # Build the stack of transformer blocks
        # Each block: LayerNorm → Attention → LayerNorm → FFN
        # With optional: DropPath, LayerScale
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,  # Stochastic depth probability
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,  # Activation for FFN (GELU is standard for ViT)
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,  # LayerScale initial value (None to disable)
                mask_k_bias=mask_k_bias,  # For iBOT masked attention
                device=device,
            )
            for i in range(depth)
        ]

        # chunked_blocks is for memory-efficient training (not used here)
        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # =================================================================
        # Step 8: Create Output Normalization Layer(s)
        # =================================================================
        # Primary norm: Applied to all tokens, or just patch tokens when untied
        self.norm = norm_layer_cls(embed_dim)

        # Optional separate norm for CLS token (allows different statistics)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        # Optional separate norm for local crop CLS tokens during training
        # Helps when global/local crops have different feature distributions
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None

        # =================================================================
        # Step 9: Create Classification Head and Mask Token
        # =================================================================
        # Head: Identity by default (just returns CLS features)
        # Can be replaced with nn.Linear for classification fine-tuning
        self.head = nn.Identity()

        # Mask token: Learnable embedding that replaces masked patches
        # Used for iBOT masked image modeling loss
        # Shape [1, embed_dim] - will be broadcast to masked positions
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self) -> None:
        """Initialize all model weights.

        This method should be called after model construction to properly
        initialize all parameters. It performs:

        1. RoPE embedding weight initialization
        2. CLS token: Normal distribution with std=0.02
        3. Storage tokens: Normal distribution with std=0.02
        4. Mask token: Zeros
        5. All other layers via :func:`init_weights_vit`

        Example:
            >>> model = DinoVisionTransformer(embed_dim=768)
            >>> model.init_weights()  # Call after construction
        """
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[int, int]]:
        """Embed image patches and assemble tokens with optional masking.

        This method converts an image batch into a sequence of tokens:
        [CLS] + [Storage×R] + [Patches×P], where masked patches are replaced
        with learnable mask tokens.

        Args:
            x: Input images of shape ``[B, C, H, W]``.
            masks: Optional boolean mask of shape ``[B, num_patches]``.
                True values indicate positions to be masked (replaced with
                mask token). Used for iBOT masked image modeling.

        Returns:
            Tuple of:
                - tokens: Token sequence of shape ``[B, 1+R+P, D]`` where
                  R=n_storage_tokens and P=num_patches.
                - hw_tuple: Original patch grid dimensions ``(H_patches, W_patches)``.

        Note:
            When masks=None, a dummy operation ``cls_token + 0 * mask_token``
            is performed to ensure mask_token gradients flow during training.
        """
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        """Process multiple image crops through the transformer.

        This is the main forward method for DINOv3 training, which processes
        multiple crops (global + local) in a single forward pass for efficiency.
        Each crop can have different resolutions thanks to RoPE position encoding.

        Args:
            x_list: List of image batches, each with shape ``[B, C, H_i, W_i]``.
                Typically contains 2 global crops (224x224) and several local
                crops (96x96).
            masks_list: List of boolean masks corresponding to each crop,
                each with shape ``[B, num_patches_i]``. Used for iBOT masked
                image modeling.

        Returns:
            List of output dictionaries, one per input crop. Each dict contains:
                - ``x_norm_clstoken``: CLS token features ``[B, D]``
                - ``x_storage_tokens``: Register token features ``[B, R, D]``
                - ``x_norm_patchtokens``: Patch token features ``[B, P_i, D]``
                - ``x_prenorm``: Pre-normalization features ``[B, 1+R+P_i, D]``
                - ``masks``: Input mask for this crop

        Note:
            The transformer blocks process all crops together by maintaining
            a list of token sequences, enabling efficient batched attention.
        """
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None
    ) -> List[Dict[str, Tensor]]:
        """Extract features from image(s).

        Convenience wrapper around :meth:`forward_features_list` that handles
        both single images and lists of images.

        Args:
            x: Either a single image batch ``[B, C, H, W]`` or a list of
                image batches for multi-crop processing.
            masks: Optional mask tensor(s). For single image, shape is
                ``[B, num_patches]``. For list input, should be a list of masks.

        Returns:
            For single input: Output dict with CLS, patch, and storage tokens.
            For list input: List of output dicts, one per input crop.

        See Also:
            :meth:`forward_features_list`: Main multi-crop forward method.
        """
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: Union[int, Sequence] = 1
    ) -> List[Tensor]:
        """Extract features from intermediate transformer blocks.

        Internal method that processes a single image through the transformer
        and collects outputs from specified intermediate layers.

        Args:
            x: Input image batch of shape ``[B, C, H, W]``.
            n: Which layers to extract. If int, extracts the last n layers.
                If sequence, extracts layers at those specific indices.

        Returns:
            List of intermediate feature tensors, each with shape
            ``[B, 1+R+P, D]`` where R=n_storage_tokens and P=num_patches.
        """
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Extract intermediate layer features for downstream tasks.

        This method is useful for dense prediction tasks (segmentation,
        depth estimation) that benefit from multi-scale features from
        different transformer depths.

        Args:
            x: Input image batch of shape ``[B, C, H, W]``.
            n: Which layers to extract. If int, extracts the last n layers.
                If sequence (e.g., [6, 12, 18, 24]), extracts those specific
                layer indices (0-indexed).
            reshape: If True, reshape patch tokens to spatial format
                ``[B, D, H//patch_size, W//patch_size]``.
            return_class_token: If True, also return CLS tokens.
            return_extra_tokens: If True, also return storage/register tokens.
            norm: If True, apply layer normalization to outputs.

        Returns:
            Tuple of outputs. Shape depends on arguments:
                - Default: ``(patch_features_1, patch_features_2, ...)``
                - With return_class_token: ``((patches, cls), (patches, cls), ...)``
                - With return_extra_tokens: ``((patches, extra), ...)``
                - With both: ``((patches, cls, extra), ...)``

        Example:
            >>> # Extract last 4 layers for dense prediction
            >>> outputs = model.get_intermediate_layers(
            ...     images, n=4, reshape=True
            ... )
            >>> # Each output has shape [B, D, H//16, W//16]
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(
        self, *args, is_training: bool = False, **kwargs
    ) -> List[Dict[str, Tensor]] | Tensor:
        """Forward pass through the model.

        Args:
            *args: Positional arguments passed to :meth:`forward_features`.
                Typically the input image(s) and optional masks.
            is_training: If True, returns full feature dict for SSL training.
                If False, returns only classification logits.
            **kwargs: Keyword arguments passed to :meth:`forward_features`.

        Returns:
            If is_training=True: Dict with CLS, patch, and storage tokens
                (see :meth:`forward_features_list` for details).
            If is_training=False: Classification logits from the head,
                shape ``[B, num_classes]``.

        Example:
            >>> # Training mode - get all features
            >>> output = model(images, is_training=True)
            >>> cls_features = output["x_norm_clstoken"]
            >>>
            >>> # Inference mode - get class predictions
            >>> logits = model(images, is_training=False)
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    """Create a ViT-Small model (~22M parameters).

    Configuration: 384-dim, 12 layers, 6 heads, FFN ratio 4.

    Args:
        patch_size: Size of image patches. Default: 16.
        **kwargs: Additional arguments passed to :class:`DinoVisionTransformer`.

    Returns:
        Uninitialized ViT-Small model. Call ``model.init_weights()`` after creation.

    Example:
        >>> model = vit_small(drop_path_rate=0.1)
        >>> model.init_weights()
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    """Create a ViT-Base model (~86M parameters).

    Configuration: 768-dim, 12 layers, 12 heads, FFN ratio 4.

    Args:
        patch_size: Size of image patches. Default: 16.
        **kwargs: Additional arguments passed to :class:`DinoVisionTransformer`.

    Returns:
        Uninitialized ViT-Base model. Call ``model.init_weights()`` after creation.

    Example:
        >>> model = vit_base(drop_path_rate=0.1)
        >>> model.init_weights()
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    """Create a ViT-Large model (~307M parameters).

    Configuration: 1024-dim, 24 layers, 16 heads, FFN ratio 4.
    This is the most commonly used variant for DINOv3 pretraining.

    Args:
        patch_size: Size of image patches. Default: 16.
        **kwargs: Additional arguments passed to :class:`DinoVisionTransformer`.

    Returns:
        Uninitialized ViT-Large model. Call ``model.init_weights()`` after creation.

    Example:
        >>> model = vit_large(
        ...     drop_path_rate=0.3,
        ...     layerscale_init=1e-5,
        ... )
        >>> model.init_weights()
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    """Create a ViT-SO400M model (~400M parameters).

    Configuration: 1152-dim, 27 layers, 18 heads, FFN ratio 3.78.
    "SO" stands for "Somewhat Optimized" - a balanced larger variant.

    Args:
        patch_size: Size of image patches. Default: 16.
        **kwargs: Additional arguments passed to :class:`DinoVisionTransformer`.

    Returns:
        Uninitialized ViT-SO400M model. Call ``model.init_weights()`` after creation.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    """Create a ViT-Huge2 model (~632M parameters).

    Configuration: 1280-dim, 32 layers, 20 heads, FFN ratio 4.

    Args:
        patch_size: Size of image patches. Default: 16.
        **kwargs: Additional arguments passed to :class:`DinoVisionTransformer`.

    Returns:
        Uninitialized ViT-Huge2 model. Call ``model.init_weights()`` after creation.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    """Create a ViT-Giant2 model (~1.1B parameters).

    Configuration: 1536-dim, 40 layers, 24 heads, FFN ratio 4.
    Close to ViT-giant, with embed-dim per head = 64.

    Args:
        patch_size: Size of image patches. Default: 16.
        **kwargs: Additional arguments passed to :class:`DinoVisionTransformer`.

    Returns:
        Uninitialized ViT-Giant2 model. Call ``model.init_weights()`` after creation.

    Note:
        This is a very large model requiring significant GPU memory.
        Consider using FSDP for training.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    """Create a ViT-7B model (~7 billion parameters).

    Configuration: 4096-dim, 40 layers, 32 heads, FFN ratio 3.
    This is the largest supported model variant.

    Args:
        patch_size: Size of image patches. Default: 16.
        **kwargs: Additional arguments passed to :class:`DinoVisionTransformer`.

    Returns:
        Uninitialized ViT-7B model. Call ``model.init_weights()`` after creation.

    Warning:
        This model requires multi-GPU training with FSDP. Training on a single
        GPU is not practical due to memory requirements.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
