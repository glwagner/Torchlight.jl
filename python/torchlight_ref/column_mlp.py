"""Paper-inspired column MLP (Farchi et al. 2025, QJRMS, doi:10.1002/qj.4934).

This is an *independent, paper-inspired* reconstruction, not the authors'
code.  Architecture: widths 420 -> 512 -> 512 -> 512 -> 512 -> 412, tanh
hidden activations, linear output, dropout p = 0.1 after each hidden
activation during training.  The activation and dropout are applied
functionally inside ``forward`` on purpose: a module listing alone does not
reveal the executed computation, which is one of the diagnosis questions the
project studies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PAPER_WIDTHS = (420, 512, 512, 512, 512, 412)
TINY_WIDTHS = (7, 11, 5, 3)


class ColumnMLP(nn.Module):
    def __init__(self, widths=PAPER_WIDTHS, dropout_p: float = 0.1):
        super().__init__()
        widths = tuple(int(w) for w in widths)
        if len(widths) < 2:
            raise ValueError("widths needs at least an input and an output size")
        self.widths = widths
        self.dropout_p = float(dropout_p)
        if not (0.0 <= self.dropout_p < 1.0):
            raise ValueError(f"dropout_p must satisfy 0 <= p < 1, got {dropout_p}")
        self.layers = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
        )

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def layer_keys(self):
        return [f"layers.{i}" for i in range(self.n_layers)]

    def forward(self, x: torch.Tensor, masks=None):
        """Forward pass.

        ``masks`` optionally supplies one 0/1 dropout mask per hidden layer
        (already *unscaled*; scaling by 1/(1-p) happens here).  When ``masks``
        is None and ``self.training`` is True, ``F.dropout`` draws masks from
        the global torch RNG.  Returns ``(y, intermediates)`` where
        ``intermediates`` is a dict of named tensors.
        """
        if masks is not None:
            self._check_masks(x, masks)
        inter = {}
        h = x
        last = self.n_layers - 1
        for i, layer in enumerate(self.layers):
            z = layer(h)
            inter[f"layer{i}_preact"] = z
            if i == last:
                h = z
                break
            a = torch.tanh(z)
            inter[f"layer{i}_act"] = a
            if masks is not None:
                m = masks[i].to(a.dtype)
                a = a * m / (1.0 - self.dropout_p)
                inter[f"layer{i}_drop"] = a
            elif self.training and self.dropout_p > 0:
                a = F.dropout(a, p=self.dropout_p, training=True)
                inter[f"layer{i}_drop"] = a
            h = a
        return h, inter


    def _check_masks(self, x, masks):
        """Explicit masks are a train-mode diagnostic; reject anything else."""
        if not self.training:
            raise ValueError("explicit dropout masks are only valid in train mode")
        n_hidden = self.n_layers - 1
        if len(masks) != n_hidden:
            raise ValueError(f"expected {n_hidden} masks, got {len(masks)}")
        for i, m in enumerate(masks):
            expected = (x.shape[0], self.widths[i + 1])
            if tuple(m.shape) != expected:
                raise ValueError(f"mask {i} has shape {tuple(m.shape)}, expected {expected}")
            if not torch.all((m == 0) | (m == 1)):
                raise ValueError(f"mask {i} must contain only 0/1 values")


def forward_only(model: ColumnMLP, x: torch.Tensor, masks=None) -> torch.Tensor:
    return model(x, masks)[0]


def mse_mean(y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over *all* elements (batch * out) of the squared error."""
    return torch.mean((y - target) ** 2)
