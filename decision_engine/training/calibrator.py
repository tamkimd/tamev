# decision_engine/training/calibrator.py
"""
Post-Hoc Probability Calibration (Temperature Scaling) for TAMEV.
Finds optimal scalar temperature T* on validation set to minimize NLL and ECE
using quasi-Newton L-BFGS optimization (Guo et al., 2017).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureCalibrator:
    """
    Optimizes a scalar temperature parameter T* on validation logits to minimize NLL
    using standard quasi-Newton L-BFGS optimization.
    """

    def __init__(self, lr: float = 0.05, max_iter: int = 50):
        self.lr = lr
        self.max_iter = max_iter

    def calibrate(
        self,
        val_logits: list[torch.Tensor | np.ndarray | list[float]],
        val_targets: list[int],
    ) -> float:
        """
        Computes optimal temperature T* via L-BFGS.
        """
        if not val_logits or not val_targets:
            return 1.0

        # Convert to list of 1D torch tensors
        t_logits = [
            torch.as_tensor(item_l, dtype=torch.float32).detach()
            if not isinstance(item_l, torch.Tensor)
            else item_l.detach().float()
            for item_l in val_logits
        ]
        t_targets = torch.as_tensor(val_targets, dtype=torch.long)

        # Vectorize into a single batched tensor (N, max_k) with -1e4 padding for dynamic candidate lengths
        max_k = max(len(row) for row in t_logits)
        batched_logits = torch.full((len(t_logits), max_k), -1e4, dtype=torch.float32)
        for i, row in enumerate(t_logits):
            batched_logits[i, : len(row)] = row

        # Initialize temperature parameter T = 1.0 (log_temp = 0.0)
        log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = torch.optim.LBFGS(
            [log_temp], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            scaled = batched_logits / temp
            loss = F.cross_entropy(scaled, t_targets)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            # Fallback to Adam if L-BFGS encounters flat landscape
            optimizer_fallback = torch.optim.Adam([log_temp], lr=self.lr)
            for _ in range(self.max_iter):
                optimizer_fallback.zero_grad()
                temp = torch.exp(log_temp)
                scaled = batched_logits / temp
                loss = F.cross_entropy(scaled, t_targets)
                loss.backward()
                optimizer_fallback.step()

        best_temp = float(torch.exp(log_temp).item())
        clamped_temp = max(0.1, min(10.0, best_temp))
        return round(clamped_temp, 2)
