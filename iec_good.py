from __future__ import annotations

import torch
import torch.nn as nn


class IECGOOD(nn.Module):
    """
    Interface placeholder for the IEC-GOOD model.

    The core implementation is temporarily unavailable in the review version.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_gc_layers,
        projection_dim,
        num_regions,
        min_region_size,
        max_region_coverage,
        region_temperature,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gc_layers = num_gc_layers
        self.projection_dim = projection_dim
        self.num_regions = num_regions
        self.min_region_size = min_region_size
        self.max_region_coverage = max_region_coverage
        self.region_temperature = region_temperature

        raise RuntimeError(
            "The IEC-GOOD core model implementation is withheld during peer review."
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "The IEC-GOOD forward implementation is withheld during peer review."
        )


class IECGOODTrainer:
    """
    Interface placeholder for the IEC-GOOD training procedure.

    The optimization strategy and loss implementation are temporarily withheld
    during peer review.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device=None,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        raise RuntimeError(
            "The IEC-GOOD training implementation is withheld during peer review."
        )

    def train(self, *args, **kwargs):
        raise RuntimeError(
            "The IEC-GOOD training loop is withheld during peer review."
        )

    def pretrain(self, *args, **kwargs):
        raise RuntimeError(
            "The IEC-GOOD pretraining implementation is withheld during peer review."
        )

    def evaluate(self, *args, **kwargs):
        raise RuntimeError(
            "The IEC-GOOD evaluation implementation is withheld during peer review."
        )
