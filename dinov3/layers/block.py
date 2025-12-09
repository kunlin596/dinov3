# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Transformer Block implementations for DINOv3 Vision Transformer.

This module provides the core building blocks for the DINOv3 ViT architecture:

- :class:`SelfAttentionBlock`: Standard transformer block with self-attention and FFN
- :class:`CausalSelfAttentionBlock`: Causal (autoregressive) variant for language modeling

Architecture (SelfAttentionBlock):
---------------------------------
::

    Input x [B, N, D]
         │
         ├──────────────────────────┐
         │                          │ (residual)
         ▼                          │
    ┌─────────┐                     │
    │  Norm1  │  LayerNorm          │
    └────┬────┘                     │
         │                          │
         ▼                          │
    ┌─────────┐                     │
    │  Attn   │  Self-Attention     │
    │  +RoPE  │  with optional RoPE │
    └────┬────┘                     │
         │                          │
         ▼                          │
    ┌─────────┐                     │
    │   LS1   │  LayerScale (opt)   │
    └────┬────┘                     │
         │                          │
         ▼                          │
        (+)◄────────────────────────┘
         │
         ├──────────────────────────┐
         │                          │ (residual)
         ▼                          │
    ┌─────────┐                     │
    │  Norm2  │  LayerNorm          │
    └────┬────┘                     │
         │                          │
         ▼                          │
    ┌─────────┐                     │
    │   FFN   │  MLP or SwiGLU      │
    └────┬────┘                     │
         │                          │
         ▼                          │
    ┌─────────┐                     │
    │   LS2   │  LayerScale (opt)   │
    └────┬────┘                     │
         │                          │
         ▼                          │
        (+)◄────────────────────────┘
         │
         ▼
    Output [B, N, D]

Key Features:
------------
- **Stochastic Depth (DropPath)**: Randomly drops residual branches during training
- **LayerScale**: Learnable per-channel scaling of residual outputs
- **RoPE Support**: Rotary position embeddings applied in attention
- **List Processing**: Efficient batched processing of multi-crop inputs

Example:
-------
.. code-block:: python

    from dinov3.layers.block import SelfAttentionBlock

    block = SelfAttentionBlock(
        dim=768,
        num_heads=12,
        ffn_ratio=4.0,
        drop_path=0.1,
        init_values=1e-5,  # LayerScale
    )

    x = torch.randn(4, 197, 768)  # [batch, tokens, dim]
    output = block(x)  # Same shape

See Also:
--------
- :mod:`dinov3.layers.attention`: Self-attention implementations
- :mod:`dinov3.layers.ffn_layers`: Feed-forward network implementations
- :class:`dinov3.models.vision_transformer.DinoVisionTransformer`: Uses these blocks
"""

from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from dinov3.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale  # , DropPath

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network.

    This is the fundamental building block of the DINOv3 Vision Transformer.
    Each block applies self-attention followed by a feed-forward network,
    with residual connections around both. Optionally includes LayerScale
    and stochastic depth (DropPath) for training stability and regularization.

    The block supports processing multiple input tensors simultaneously
    (for multi-crop training) via the list-based forward methods, which
    concatenate inputs for efficient batched computation.

    Args:
        dim: Input and output embedding dimension.
        num_heads: Number of attention heads. Must evenly divide ``dim``.
        ffn_ratio: Expansion ratio for FFN hidden dimension.
            FFN hidden dim = ``dim * ffn_ratio``. Default: 4.0.
        qkv_bias: Include bias in Query/Key/Value projections. Default: False.
        proj_bias: Include bias in attention output projection. Default: True.
        ffn_bias: Include bias in FFN linear layers. Default: True.
        drop: Dropout probability for FFN and attention projection. Default: 0.0.
        attn_drop: Dropout probability for attention weights. Default: 0.0.
        init_values: Initial value for LayerScale. If None, LayerScale is
            disabled (replaced with Identity). Typical values: 1e-4 to 1e-6.
            Default: None.
        drop_path: Stochastic depth probability. During training, randomly
            samples a subset of batch elements to apply the residual to.
            Provides regularization for deep networks. Default: 0.0.
        act_layer: Activation function class for FFN. Default: nn.GELU.
        norm_layer: Normalization layer class. Default: nn.LayerNorm.
        attn_class: Self-attention implementation class. Default: SelfAttention.
        ffn_layer: Feed-forward network class. Default: Mlp.
        mask_k_bias: Use learnable bias for masked key positions (for iBOT).
            Default: False.
        device: Device to create parameters on. Default: None.

    Attributes:
        norm1: Pre-attention normalization layer.
        attn: Self-attention module.
        ls1: LayerScale for attention output (or Identity if disabled).
        norm2: Pre-FFN normalization layer.
        mlp: Feed-forward network module.
        ls2: LayerScale for FFN output (or Identity if disabled).
        sample_drop_ratio: Stochastic depth probability.

    Example:
        >>> block = SelfAttentionBlock(
        ...     dim=768,
        ...     num_heads=12,
        ...     ffn_ratio=4.0,
        ...     drop_path=0.1,
        ...     init_values=1e-5,
        ... )
        >>> x = torch.randn(4, 197, 768)
        >>> output = block(x)  # [4, 197, 768]
        >>>
        >>> # Multi-crop processing
        >>> crops = [torch.randn(4, 197, 768), torch.randn(4, 49, 768)]
        >>> outputs = block(crops)  # List of 2 tensors
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the SelfAttentionBlock.

        Creates all sub-modules: normalization layers, attention, FFN,
        and optional LayerScale modules.
        """
        super().__init__()

        # =====================================================================
        # Attention Branch: Norm1 → Attention → LayerScale1
        # =====================================================================
        # Pre-norm architecture: normalize before attention
        self.norm1 = norm_layer(dim)

        # Self-attention with optional RoPE position encoding
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,  # For iBOT masked attention
            device=device,
        )

        # LayerScale: learnable per-channel scaling initialized to small values
        # Helps training stability for deep networks by starting with small residuals
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        # =====================================================================
        # FFN Branch: Norm2 → MLP/SwiGLU → LayerScale2
        # =====================================================================
        # Pre-norm architecture: normalize before FFN
        self.norm2 = norm_layer(dim)

        # Feed-forward network with expansion ratio
        # Hidden dim is typically 4x the embedding dim
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )

        # LayerScale for FFN output
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        # =====================================================================
        # Stochastic Depth (DropPath)
        # =====================================================================
        # During training, randomly drops residual branches for regularization
        # Implemented as sampling a subset of batch elements
        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(
        rope: tuple[Tensor, Tensor] | None, indices: Tensor
    ) -> tuple[Tensor, Tensor] | None:
        """Index into RoPE embeddings when using stochastic depth.

        When stochastic depth randomly samples a subset of batch elements,
        the corresponding RoPE embeddings must also be indexed if they
        have a batch dimension.

        Args:
            rope: Tuple of (sin, cos) RoPE embeddings, or None.
                Shape can be:
                - ``[batch, heads, patches, dim]``: Per-sample RoPE (needs indexing)
                - ``[heads, patches, dim]`` or ``[patches, dim]``: Shared RoPE
            indices: Batch indices selected by stochastic depth.

        Returns:
            Indexed RoPE tuple if input has batch dimension, otherwise
            returns the original tuple unchanged. Returns None if input is None.
        """
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Forward pass for a single input tensor (reference implementation).

        This is the reference implementation showing the standard transformer
        block computation. In practice, :meth:`_forward_list` is used even
        for single inputs to maintain consistency.

        The computation flow is:
            1. x_attn = x + LayerScale1(Attention(Norm1(x)))
            2. x_out = x_attn + LayerScale2(FFN(Norm2(x_attn)))

        With stochastic depth enabled during training, only a random subset
        of batch elements receive the residual updates, with appropriate
        scaling to maintain expected values.

        Args:
            x: Input tensor of shape ``[batch, tokens, dim]``.
            rope: Optional tuple of (sin, cos) RoPE embeddings for
                position-aware attention.

        Returns:
            Output tensor of same shape as input ``[batch, tokens, dim]``.

        Note:
            This method is not called directly; :meth:`forward` routes
            to :meth:`_forward_list` for both single and list inputs.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(
        self, x_list: List[Tensor], rope_list: List[tuple[Tensor, Tensor] | None] | None = None
    ) -> List[Tensor]:
        """Forward pass for multiple input tensors (multi-crop processing).

        This is the main forward implementation that efficiently processes
        multiple inputs (e.g., global and local crops) by concatenating them
        for shared elementwise operations, then splitting back.

        The key optimization is that normalization and other elementwise ops
        are applied to the concatenated tensor, reducing kernel launch overhead.
        torch.compile's memory planning hides the concat/split overhead.

        Args:
            x_list: List of input tensors, each with shape ``[B, N_i, D]``.
                Different tensors can have different sequence lengths (N_i)
                for multi-scale crop processing.
            rope_list: Optional list of RoPE tuples, one per input tensor.
                Each tuple is (sin, cos) for position encoding. If None,
                no position encoding is applied.

        Returns:
            List of output tensors, same shapes as inputs.

        Note:
            When stochastic depth is enabled, each input tensor has its own
            random subset of batch elements selected, maintaining proper
            regularization across different crop scales.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(
        self,
        x_or_x_list: Tensor | List[Tensor],
        rope_or_rope_list: tuple[Tensor, Tensor] | List[tuple[Tensor, Tensor] | None] | None = None,
    ) -> Tensor | List[Tensor]:
        """Forward pass supporting both single tensor and multi-crop list inputs.

        This is the main entry point for the transformer block. It automatically
        dispatches to the appropriate implementation based on input type.

        Args:
            x_or_x_list: Either:
                - Single tensor ``[B, N, D]`` for standard forward pass
                - List of tensors for multi-crop processing (DINOv3 training)
            rope_or_rope_list: Optional RoPE position encodings:
                - Single (sin, cos) tuple for single tensor input
                - List of tuples for multi-crop input
                - None to disable position encoding

        Returns:
            - Single tensor ``[B, N, D]`` if input was a single tensor
            - List of tensors if input was a list

        Example:
            >>> block = SelfAttentionBlock(dim=768, num_heads=12)
            >>>
            >>> # Single input
            >>> x = torch.randn(4, 197, 768)
            >>> out = block(x)  # [4, 197, 768]
            >>>
            >>> # Multi-crop input (2 global + 8 local crops)
            >>> global_crops = [torch.randn(4, 197, 768) for _ in range(2)]
            >>> local_crops = [torch.randn(4, 49, 768) for _ in range(8)]
            >>> crops = global_crops + local_crops
            >>> outputs = block(crops)  # List of 10 tensors

        Raises:
            AssertionError: If input is neither a Tensor nor a list.
        """
        if isinstance(x_or_x_list, Tensor):
            # Single tensor: wrap in list, process, unwrap
            # Uses list implementation for consistency
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            # Multi-crop list: process all together
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError(f"Expected Tensor or List[Tensor], got {type(x_or_x_list)}")


class CausalSelfAttentionBlock(nn.Module):
    """Transformer block with causal (autoregressive) self-attention.

    This block is designed for autoregressive generation tasks where each
    token can only attend to previous tokens (and itself). The causal mask
    prevents information flow from future positions.

    Used for language modeling components in multimodal models.

    Args:
        dim: Input and output embedding dimension.
        num_heads: Number of attention heads.
        ffn_ratio: Expansion ratio for FFN hidden dimension. Default: 4.0.
        ls_init_value: Initial value for LayerScale. None to disable. Default: None.
        is_causal: Whether to apply causal masking. Default: True.
        act_layer: Activation function class for FFN. Default: nn.GELU.
        norm_layer: Normalization layer class. Default: nn.LayerNorm.
        dropout_prob: Dropout probability for attention and FFN. Default: 0.0.

    Attributes:
        dim: Embedding dimension.
        is_causal: Whether causal masking is enabled.
        ls1: LayerScale for attention output.
        attention_norm: Pre-attention normalization.
        attention: Causal self-attention module.
        ffn_norm: Pre-FFN normalization.
        feed_forward: Feed-forward network.
        ls2: LayerScale for FFN output.

    Example:
        >>> block = CausalSelfAttentionBlock(
        ...     dim=768,
        ...     num_heads=12,
        ...     is_causal=True,
        ... )
        >>> x = torch.randn(4, 100, 768)  # [batch, seq_len, dim]
        >>> output = block(x)  # [4, 100, 768]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        """Initialize the CausalSelfAttentionBlock."""
        super().__init__()

        # Store architecture config
        self.dim = dim
        self.is_causal = is_causal

        # =====================================================================
        # Attention Branch: LayerScale1 → Norm → CausalAttention
        # =====================================================================
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        # Causal attention: each position attends only to previous positions
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        # =====================================================================
        # FFN Branch: Norm → MLP → LayerScale2
        # =====================================================================
        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        """Initialize weights with scaled initialization.

        Uses GPT-style initialization where output projections are scaled
        by a factor to prevent gradient explosion in deep networks.

        Args:
            init_attn_std: Std for attention QKV weights. Default: dim^-0.5.
            init_proj_std: Std for output projections. Default: init_attn_std * factor.
            init_fc_std: Std for FFN fc1 weights. Default: (2*dim)^-0.5.
            factor: Scaling factor for projection std. Typically 1/sqrt(2*depth).
                Default: 1.0.
        """
        # Default initialization scales
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5

        # Initialize attention weights
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()

        # Initialize FFN weights
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal self-attention.

        Args:
            x: Input tensor of shape ``[batch, seq_len, dim]``.

        Returns:
            Output tensor of same shape ``[batch, seq_len, dim]``.
            Each position only attends to previous positions.
        """
        # Attention with causal mask: each token attends only to past tokens
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        # Feed-forward network
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn
