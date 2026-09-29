# decision_engine/training/losses.py
"""
Composite Multi-Objective Loss for TAMEV Decision Engine.
Combines:
1. Hard-label Cross-Entropy (Task Accuracy)
2. Proper Scoring Brier Loss (Calibration & Uncertainty Penalization)
3. Direct Preference Margin Loss (Hard Negative Contrastive Separation)
4. Knowledge Distillation KL Divergence (Teacher Soft Targets)
5. Policy Gradient Brier Advantage Loss (RL Alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_padding_options(
    logits: torch.Tensor,
    num_options: list[int] | torch.Tensor | None,
    pad_value: float = -1e4,
) -> torch.Tensor:
    """Vectorized masking of variable-cardinality padding slots without Python loops."""
    if num_options is None:
        return logits
    _, K = logits.shape
    if isinstance(num_options, list):
        n_opts = torch.tensor(num_options, device=logits.device, dtype=torch.long)
    else:
        n_opts = num_options.to(device=logits.device, dtype=torch.long)
    mask = torch.arange(K, device=logits.device).unsqueeze(0) >= n_opts.unsqueeze(1)
    return logits.masked_fill(mask, pad_value)


class CompositeLoss(nn.Module):
    """
    Unified loss function balancing task accuracy, calibration, and teacher imitation.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        brier_weight: float = 0.5,
        margin_weight: float = 0.4,
        kd_weight: float = 0.3,
        rl_weight: float = 0.05,
        margin_gamma: float = 1.2,
        kd_temp: float = 2.0,
        ordinal_weight: float = 0.0,
        cell_weights: dict[str, float] | None = None,
        rdrop_weight: float = 0.0,
        sce_weight: float = 0.0,
        sce_alpha: float = 0.1,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.brier_weight = brier_weight
        self.margin_weight = margin_weight
        self.kd_weight = kd_weight
        self.rl_weight = rl_weight
        self.margin_gamma = margin_gamma
        self.kd_temp = kd_temp
        # Ordinal cells (`decision_type: score`, e.g. 1-5 star ratings) are ordered, and the option
        # list is generated ascending by value (schema.DecisionItem.from_dict), so |i - target| is
        # the ordinal distance. A flat softmax treats "2 stars" and "5 stars" as equally wrong;
        # this term charges for the probability mass placed far from the target.
        self.ordinal_weight = ordinal_weight
        # Optional per-cell sample weights, keyed "<source>/<qid>" (e.g. "yelp/rating"). Empty/None
        # is off. The trainer resolves them per item and renormalises so the batch mean is 1.0.
        self.cell_weights = cell_weights or {}
        # R-Drop consistency weight; the second forward pass is the trainer's job, this only scores
        # the symmetric KL between the two logit sets. 0.0 keeps every path below bit-identical.
        self.rdrop_weight = rdrop_weight
        # Noise-robust symmetric cross-entropy (Wang et al. 2019, arXiv:1908.06112). Only the RCE
        # half is added here, because the CE half is already term 1: SCE = CE + sce_weight * RCE.
        # RCE = -sum_k p_k log t_k, with the truth `t` smoothed by `sce_alpha` -- a one-hot t would
        # make log t = -inf off-target. 0.0 = off, and the branch below is then never entered, so
        # every other path stays bit-identical to the pre-knob loss.
        self.sce_weight = sce_weight
        self.sce_alpha = sce_alpha
        if sce_weight > 0 and not 0.0 < sce_alpha < 1.0:
            raise ValueError(f"sce_weight>0 needs sce_alpha in (0,1); got {sce_alpha}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
        num_options: list[int] | None = None,
        has_teacher: torch.Tensor | None = None,
        ref_logits: torch.Tensor | None = None,
        ordinal: torch.Tensor | None = None,
        cells: list[str] | None = None,
        logits2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        logits: (B, K)
        targets: (B,)
        teacher_probs: (B, K) or None
        num_options: List of valid options count per item [K_1, ..., K_B]
        has_teacher: (B,) boolean tensor indicating whether teacher_probs is valid
        ref_logits: (B, K) or None, logits from frozen reference anchor model
        ordinal: (B,) boolean tensor marking ordered-option items; only scored when
            `ordinal_weight > 0`. `targets` is the target option *index* and the option axis is
            ascending in value for those items, so the distance is |index - target|.
        cells: (B,) cell label per item ("<source>/<qid>"), used only to look up
            `cell_weights`; `None` or an empty weight table is a no-op.
        """
        B, K = logits.shape
        device = logits.device

        # Mask invalid options so padding options do not participate (vectorized)
        masked_logits = _mask_padding_options(logits, num_options)

        log_p = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)

        # 1. Base Task Cross-Entropy Loss
        ce_loss = F.nll_loss(log_p, targets)

        # 2. Proper Scoring Brier Loss
        one_hot = F.one_hot(targets, num_classes=K).float()
        brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))

        # 3. Direct Preference Margin Loss (Contrastive Hard Negative Separation)
        # s_w/s_l are computed on both paths so the cell-weighted block below has them bound; at K == 1
        # they are unused (margin_loss is zero), and the cost is one gather over a single column.
        neg_logits = masked_logits.clone()
        neg_logits[torch.arange(B, device=device), targets] = -1e4
        s_l = torch.max(neg_logits, dim=-1).values
        s_w = masked_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        margin_loss = (
            torch.mean(F.relu(self.margin_gamma - (s_w - s_l)))
            if K > 1
            else torch.tensor(0.0, device=device)
        )

        # 4. Soft Label Distillation (KL Divergence against Teacher or Reference Anchor)
        log_student_kd = F.log_softmax(masked_logits / self.kd_temp, dim=-1)
        if teacher_probs is not None and has_teacher is not None and has_teacher.any():
            if ref_logits is not None:
                ref_masked = _mask_padding_options(ref_logits, num_options)
                ref_probs_kd = F.softmax(ref_masked / self.kd_temp, dim=-1)
                kd_targets = ref_probs_kd.clone()
                kd_targets[has_teacher] = teacher_probs[has_teacher]
                kd_loss = -torch.mean(torch.sum(kd_targets.detach() * log_student_kd, dim=-1))
            else:
                kd_loss = -torch.mean(
                    torch.sum(
                        teacher_probs[has_teacher].detach() * log_student_kd[has_teacher], dim=-1
                    )
                )
        elif ref_logits is not None:
            ref_masked = _mask_padding_options(ref_logits, num_options)
            ref_probs_kd = F.softmax(ref_masked / self.kd_temp, dim=-1)
            kd_loss = -torch.mean(torch.sum(ref_probs_kd.detach() * log_student_kd, dim=-1))
        elif teacher_probs is not None and has_teacher is None:
            kd_loss = -torch.mean(torch.sum(teacher_probs.detach() * log_student_kd, dim=-1))
        else:
            kd_loss = torch.tensor(0.0, device=device)

        # 5. Reinforcement Learning: Proper Scoring Brier Reward Policy Gradient
        brier_rewards = 1.0 - torch.sum((probs - one_hot) ** 2, dim=-1)
        advantage = brier_rewards - brier_rewards.mean()
        selected_log_p = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        rl_loss = -torch.mean(advantage.detach() * selected_log_p)

        # 6. Ordinal distance penalty (score cells only)
        ordinal_loss = torch.tensor(0.0, device=device)
        if self.ordinal_weight > 0 and ordinal is not None and bool(ordinal.any()):
            idx = torch.arange(K, device=device, dtype=probs.dtype)
            distance = (idx.unsqueeze(0) - targets.unsqueeze(1)).abs()
            ordinal_loss = torch.mean(torch.sum(probs * distance, dim=-1)[ordinal])

        # 6b. R-Drop consistency (symmetric KL between two dropout-masked forward passes).
        # Unweighted by `cell_weights` for the same reason as KD: it is a whole-batch mean, so a
        # per-item multiplier would silently change the reduction. Padding slots carry
        # `pad_value = -1e4`, so their softmax mass is ~0 and they drop out of the sum: the term
        # depends only on the valid options of each row, never on their order.
        rdrop_loss = torch.tensor(0.0, device=device)
        if self.rdrop_weight > 0 and logits2 is not None:
            masked2 = _mask_padding_options(logits2, num_options)
            log_p2 = F.log_softmax(masked2, dim=-1)
            p2 = F.softmax(masked2, dim=-1)
            kl_12 = torch.sum(probs * (log_p - log_p2), dim=-1)
            kl_21 = torch.sum(p2 * (log_p2 - log_p), dim=-1)
            rdrop_loss = 0.5 * torch.mean(kl_12 + kl_21)

        # 6c. Noise-robust reverse cross-entropy (the RCE half of SCE; see __init__). `t` is the
        # truth smoothed by `sce_alpha`, so only the target column differs from a constant: the term
        # is invariant under option permutation, and padding slots carry p ~ 0 (masked) so they drop
        # out. Kept per-row so the cell-weighted path below can weight it like CE.
        rce_items = None
        if self.sce_weight > 0:
            t = one_hot * (1.0 - self.sce_alpha) + self.sce_alpha / K
            rce_items = -torch.sum(probs * torch.log(t), dim=-1)
        sce_loss = rce_items.mean() if rce_items is not None else torch.tensor(0.0, device=device)

        # 7. Optional per-cell sample weighting (`training.cell_weights`). Weight is a per-item
        # scalar looked up from the item's cell label and renormalised to a batch mean of exactly
        # 1.0 -- without that, a table of all-2.0 would silently double the effective LR instead of
        # only moving where the loss is spent. Weighted: the per-item accuracy/calibration terms
        # CE, Brier, margin and SCE. Not weighted: KD (an explicitly masked-subset mean), RL (its
        # advantage is already batch-mean-centred) and ordinal (score-subset mean) -- a per-item
        # multiplier is not the same reduction as those terms implement, and the ask targets the
        # error-bearing terms. A weight depends only on the item, never on option order or K, so it
        # cannot change under option permutation: option-permutation drift stays exactly 0.
        # `cells=None` or an empty table keeps the pre-knob expressions bit for bit.
        if self.cell_weights and cells is not None:
            if len(cells) != B:
                raise ValueError(f"cells length {len(cells)} != batch size {B}")
            w = torch.tensor(
                [self.cell_weights.get(str(c), 1.0) for c in cells],
                device=device,
                dtype=logits.dtype,
            )
            mean_w = w.mean()
            if not mean_w > 0:
                raise ValueError("cell_weights must give this batch a positive mean weight")
            w = w / mean_w
            ce_loss = (-log_p.gather(1, targets.unsqueeze(1)).squeeze(1) * w).mean()
            brier_loss = (torch.sum((probs - one_hot) ** 2, dim=-1) * w).mean()
            if K > 1:
                margin_loss = (F.relu(self.margin_gamma - (s_w - s_l)) * w).mean()
            if rce_items is not None:
                sce_loss = (rce_items * w).mean()

        # Total Composite Multi-Objective Loss
        total_loss = (
            self.ce_weight * ce_loss
            + self.brier_weight * brier_loss
            + self.margin_weight * margin_loss
            + self.kd_weight * kd_loss
            + self.rl_weight * rl_loss
            + self.ordinal_weight * ordinal_loss
            + self.rdrop_weight * rdrop_loss
            + self.sce_weight * sce_loss
        )

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "brier_loss": brier_loss,
            "margin_loss": margin_loss,
            "kd_loss": kd_loss,
            "rl_loss": rl_loss,
            "ordinal_loss": ordinal_loss,
            "rdrop_loss": rdrop_loss,
            "sce_loss": sce_loss,
        }


def compute_brier_reward(pred_prob: float, is_correct: bool) -> float:
    """Proper Brier scoring reward: R = 1.0 - (prob - y)^2."""
    y = 1.0 if is_correct else 0.0
    return 1.0 - (pred_prob - y) ** 2
