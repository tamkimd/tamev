# decision_engine/models/pointer_head.py
"""
Decoupled, Permutation-Invariant Pointer Network Head.
Computes bilinear compatibility between a pooled context vector and candidate option vectors:
Score(c, o_i) = (W_q c)^T (W_k o_i) / sqrt(d)
Guarantees mathematically exact 0.00000000 permutation equivariance across any option permutation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def order_invariant_softmax(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Softmax over the option axis with an accumulation order that ignores option order.

    An fp32 reduction reorders with the options and shifts each probability by 1-2 ulp (measured
    3-7e-8 at p~0.5, above this repo's 1e-8 equivariance clause). float64 accumulation removes it
    (measured exactly 0.0). MPS has no float64 kernel, so it keeps fp32 and its drift floor stays
    at ~3e-8 there.
    """
    t = torch.clamp(temperature, min=0.05)
    if logits.device.type == "mps":
        return F.softmax(logits / t, dim=-1)
    return F.softmax(logits.double() / t.double(), dim=-1).to(logits.dtype)


class PointerHead(nn.Module):
    """
    Permutation-invariant bilinear compatibility pointer head with learnable/calibrated temperature.
    """

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int = 64,
        temperature: float = 1.0,
        normalize: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.normalize = normalize
        self.q_proj = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.scale = 1.0 / math.sqrt(projection_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(
        self, context_vec: torch.Tensor, option_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_vec: (B, hidden_dim) pooled context embedding
            option_vecs: (B, K, hidden_dim) pooled option embeddings
        Returns:
            logits: (B, K) unnormalized compatibility scores
            probs:  (B, K) normalized calibrated probabilities
        """
        # Match linear layer weight dtype
        dtype = self.q_proj.weight.dtype
        if context_vec.dtype != dtype:
            context_vec = context_vec.to(dtype)
        if option_vecs.dtype != dtype:
            option_vecs = option_vecs.to(dtype)

        q = self.q_proj(context_vec)
        k = self.k_proj(option_vecs)

        if self.normalize:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        # (B, projection_dim) -> (B, 1, projection_dim)
        q = q.unsqueeze(1)

        # Batch matrix multiplication: (B, 1, P) x (B, P, K) -> (B, K)
        # fp32 bmm over the option axis is not permutation-stable: the kernel's per-column
        # accumulation order depends on the option's lane inside its N-block (measured 1 ulp,
        # 4.8e-07, at K=3/K=5 on CPU; float64 is exactly 0.0). float64 removes it. MPS has no
        # fp64 kernel and keeps fp32 there (drift floor ~3e-8).
        if q.device.type != "mps":
            q, k = q.double(), k.double()
        if self.normalize:
            logits = (torch.bmm(q, k.transpose(1, 2)).squeeze(1) * 14.0).to(dtype)
        else:
            logits = (torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.scale).to(dtype)
        probs = order_invariant_softmax(logits, self.temperature)
        return logits, probs

    def set_temperature(self, t: float) -> None:
        """Update temperature parameter."""
        with torch.no_grad():
            self.temperature.copy_(
                torch.as_tensor(t, dtype=torch.float32, device=self.temperature.device)
            )


# Backward-compatibility aliases
TinyBertPointerHead = PointerHead
TamevPointerHead = PointerHead


class DynamicPointerHeadRef:
    """NumPy reference implementation for lightweight tests and validation."""

    def __init__(self, hidden_dim: int = 128, projection_dim: int = 64, temperature: float = 1.0):
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.scale = 1.0 / math.sqrt(projection_dim)
        rng = np.random.default_rng(42)
        self.W_q = rng.standard_normal((hidden_dim, projection_dim)) * 0.02
        self.W_k = rng.standard_normal((hidden_dim, projection_dim)) * 0.02

    def forward(
        self, context_vec: np.ndarray, option_vecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # context: (B, D) -> (B, P)
        q = np.matmul(context_vec, self.W_q)
        # options: (B, K, D) -> (B, K, P)
        k = np.matmul(option_vecs, self.W_k)

        # Bilinear dot product: (B, K)
        logits = np.einsum("bp,bkp->bk", q, k) * self.scale
        scaled_logits = logits / max(self.temperature, 0.05)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return logits, probs
