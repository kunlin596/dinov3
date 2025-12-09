# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Meta Data Loaders for Multi-Dataset Training.

This module provides utilities for combining multiple PyTorch DataLoaders into
a single unified iterator, enabling training on heterogeneous data sources with
controlled sampling ratios.
# Type alias for a data loader that yields batches (lists of samples)
Loader = Iterable[list[Any]]


# ==============================================================================
# Combined Data Loader
# ==============================================================================
Key Concepts:
------------
**Multi-Dataset Training**
    DINOv3 can train on multiple datasets simultaneously (e.g., ImageNet + custom data).
    This module handles the orchestration of sampling from each dataset according
    to specified ratios.

**Sampling Modes**
    - ``GLOBAL_HOMOGENEOUS``: Deterministic sampling with fixed seed for reproducibility.
      Each rank samples the same sequence of dataset choices.
    - ``LOCAL_HOMOGENEOUS``: Random sampling per rank. Different ranks may sample
      different datasets at each iteration.

Architecture:
------------
::

    ┌─────────────────────────────────────────────────────────────┐
    │                   CombinedDataLoader                        │
    │                                                             │
    │   ┌───────────┐  ┌───────────┐  ┌───────────┐              │
    │   │ Loader 1  │  │ Loader 2  │  │ Loader 3  │   ...        │
    │   │ (ratio=   │  │ (ratio=   │  │ (ratio=   │              │
    │   │   0.7)    │  │   0.2)    │  │   0.1)    │              │
    │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘              │
    │         │              │              │                     │
    │         └──────────────┼──────────────┘                     │
    │                        │                                    │
    │                   ┌────▼────┐                               │
    │                   │   RNG   │  Probabilistic selection      │
    │                   │ choice  │  based on ratios              │
    │                   └────┬────┘                               │
    │                        │                                    │
    │                   ┌────▼────┐                               │
    │                   │  Batch  │  Single batch from            │
    │                   │ Output  │  selected loader              │
    │                   └─────────┘                               │
    └─────────────────────────────────────────────────────────────┘

Usage Example:
-------------
.. code-block:: python

    from dinov3.data.meta_loaders import CombinedDataLoader

    # Create individual data loaders
    imagenet_loader = DataLoader(imagenet_dataset, batch_size=64)
    custom_loader = DataLoader(custom_dataset, batch_size=64)

    # Combine with 80% ImageNet, 20% custom
    combined = CombinedDataLoader(
        loaders_with_ratios=[
            (imagenet_loader, 0.8),
            (custom_loader, 0.2),
        ],
        batch_size=64,
        combining_mode=CombinedDataLoader.GLOBAL_HOMOGENEOUS,
        seed=42,
        name="train_combined",
    )

    for batch in combined:
        # Each batch comes from one of the loaders
        # Probability of ImageNet batch: 80%
        # Probability of custom batch: 20%
        train_step(batch)

Note:
----
All loaders must have the same batch size to ensure consistent gradient
accumulation and memory usage across iterations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from collections.abc import Iterable, Iterator

logger = logging.getLogger("dinov3")

# Type alias for a data loader that yields batches (lists of samples)
Loader = Iterable[list[Any]]


class CombinedDataLoader:
    """
    Combines multiple data loaders with probabilistic sampling.

    This class wraps multiple PyTorch DataLoaders and yields batches from them
    according to specified sampling ratios. At each iteration, one loader is
    selected probabilistically, and a batch is drawn from it.

    This enables training on multiple datasets simultaneously with controlled
    data mixing ratios, which is useful for:

    - **Domain adaptation**: Mix source and target domain data
    - **Data augmentation**: Combine real and synthetic data
    - **Multi-task learning**: Sample from task-specific datasets
    - **Curriculum learning**: Gradually shift ratios during training

    Attributes:
    ----------
    GLOBAL_HOMOGENEOUS : int
        Mode constant (0). Uses deterministic seed for reproducible sampling.
        All ranks in distributed training will make the same sequence of
        loader choices (though they may see different data within each loader
        due to distributed samplers).

    LOCAL_HOMOGENEOUS : int
        Mode constant (1). Uses random seed for non-deterministic sampling.
        Each rank independently chooses which loader to sample from.

    loaders : tuple[Loader, ...]
        The underlying data loaders being combined.

    ratios : tuple[float, ...]
        Sampling probabilities for each loader (must sum to 1.0).

    batch_size : int
        Batch size (must be identical across all loaders).

    loader_count : np.ndarray
        Running count of how many times each loader has been sampled.
        Useful for monitoring actual vs. expected ratios.

    Example:
    -------
    .. code-block:: python

        # 70% ImageNet, 30% custom dataset
        combined = CombinedDataLoader(
            loaders_with_ratios=[
                (imagenet_loader, 0.7),
                (custom_loader, 0.3),
            ],
            batch_size=64,
            combining_mode=CombinedDataLoader.GLOBAL_HOMOGENEOUS,
            seed=42,
            name="train",
            logging_period=100,
        )

        for epoch in range(num_epochs):
            for batch in combined:
                loss = model(batch)
                loss.backward()

    Warning:
    -------
    The iterator stops when ANY of the underlying loaders is exhausted.
    Ensure loaders have similar lengths or use infinite samplers if needed.
    """

    GLOBAL_HOMOGENEOUS: int = 0
    """Deterministic sampling mode with fixed seed for reproducibility."""

    LOCAL_HOMOGENEOUS: int = 1
    """Random sampling mode with per-instance random seed."""

    def __init__(
        self,
        loaders_with_ratios: Iterable[tuple[Loader, float]],
        batch_size: int,
        combining_mode: int = 1,
        seed: int = 65537,
        name: str | None = None,
        logging_period: int = 100,
    ) -> None:
        """
        Initialize the combined data loader.

        Parameters:
        ----------
        loaders_with_ratios : Iterable[tuple[Loader, float]]
            Iterable of (loader, ratio) pairs. Ratios should sum to 1.0.

            Example::

                [
                    (imagenet_loader, 0.7),  # 70% of batches
                    (custom_loader, 0.3),    # 30% of batches
                ]

        batch_size : int
            Expected batch size. All loaders must have this same batch size.
            This is validated at initialization.

        combining_mode : int, default=1
            Sampling mode:

            - ``GLOBAL_HOMOGENEOUS`` (0): Deterministic, reproducible sampling
            - ``LOCAL_HOMOGENEOUS`` (1): Random sampling per instance

        seed : int, default=65537
            Random seed for ``GLOBAL_HOMOGENEOUS`` mode.
            Ignored in ``LOCAL_HOMOGENEOUS`` mode.

        name : str | None, default=None
            Optional name for logging purposes. Helps identify which
            combined loader is reporting statistics.

        logging_period : int, default=100
            How often (in iterations) to log empirical sampling ratios.
            Set to a large number to reduce log verbosity.

        Raises:
        ------
        ValueError
            If ``combining_mode`` is not a valid mode constant.

        AssertionError
            If any loader has a different batch size.
        """
        if combining_mode not in [self.GLOBAL_HOMOGENEOUS, self.LOCAL_HOMOGENEOUS]:
            raise ValueError(f"Unsupported value of combining_mode ({combining_mode})")
        loaders, ratios = zip(*loaders_with_ratios)
        assert np.all(
            [loader.batch_size == batch_size for loader in loaders]
        ), f"All individual loaders must have the same batch size to the combined data loader for combining_mode={combining_mode}"
        self.loaders = loaders
        self.ratios = ratios
        self.batch_size = batch_size
        self.combining_mode = combining_mode
        self.initial_seed = seed
        self.name = name if name is not None else ""
        self.logging_period = logging_period
        if combining_mode == self.GLOBAL_HOMOGENEOUS:
            logger.info(f"Initialize CDL {self.name} with seed={seed}")
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)
        else:
            logger.info(f"Initialize CDL {self.name} with random seed")
            self.seed = 0
            self.rng = np.random.default_rng()
        self.loader_count = np.zeros(len(self.loaders))

    def homogeneous_iterator(self) -> Iterator[list[Any]]:
        """
        Create an iterator that samples loaders with uniform batch composition.

        "Homogeneous" means each yielded batch comes entirely from one loader
        (no mixing of samples from different loaders within a single batch).

        The iterator:

        1. Probabilistically selects a loader based on ``self.ratios``
        2. Yields the next batch from that loader
        3. Updates ``self.loader_count`` for monitoring
        4. Periodically logs empirical vs. expected ratios
        5. Stops when any loader is exhausted

        Yields:
        ------
        list[Any]
            A batch from one of the underlying loaders.

        Note:
        ----
        The empirical ratios may differ slightly from specified ratios,
        especially early in training or with short epochs. The periodic
        logging helps monitor this drift.

        Example:
        -------
        If ratios are [0.7, 0.3] for two loaders:

        - ~70% of iterations yield batches from loader 0
        - ~30% of iterations yield batches from loader 1
        - Each batch contains samples from only one dataset
        """
        iteration = 0
        iters = [iter(loader) for loader in self.loaders]
        while True:
            iteration += 1
            try:
                # Probabilistically select a loader based on ratios
                idx = self.rng.choice(len(self.loaders), p=self.ratios)
                self.loader_count[idx] += 1

                # Log empirical ratios periodically for monitoring
                if iteration % self.logging_period == 0:
                    logger.info(f"Empirical ratios: CDL {self.name} {self.loader_count / self.loader_count.sum()}")

                yield next(iters[idx])
            except StopIteration:
                # Stop when any loader is exhausted
                break

    def heterogeneous_iterator(self) -> Iterator[list[Any]]:
        """
        Create an iterator that mixes samples from different loaders within batches.

        **Not yet implemented.**

        This would create batches where samples come from multiple loaders,
        mixed according to the ratios. For example, with ratios [0.7, 0.3]
        and batch_size=64:

        - ~45 samples from loader 0
        - ~19 samples from loader 1
        - Mixed together in a single batch

        Raises:
        ------
        NotImplementedError
            This method is a placeholder for future implementation.
        """
        raise NotImplementedError("Heterogeneous iterator not yet implemented")

    def __iter__(self) -> Iterator[list[Any]]:
        """
        Return an iterator over combined batches.

        Selects the appropriate iterator based on ``combining_mode``:

        - ``GLOBAL_HOMOGENEOUS``: Deterministic homogeneous sampling
        - ``LOCAL_HOMOGENEOUS``: Random homogeneous sampling

        Returns:
        -------
        Iterator[list[Any]]
            Iterator yielding batches from the combined loaders.

        Raises:
        ------
        ValueError
            If ``combining_mode`` is not recognized.
        """
        if self.combining_mode in [self.GLOBAL_HOMOGENEOUS, self.LOCAL_HOMOGENEOUS]:
            logger.info(f"Using homogeneous iterator for CDL {self.name}")
            return self.homogeneous_iterator()
        else:
            raise ValueError(f"Unsupported value of combining_mode ({self.combining_mode})")
