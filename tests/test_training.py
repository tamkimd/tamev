"""Training: composite/Brier/RL losses, the SFT/calibration/fine-tune pipeline, early stopping and
composite scoring, the K-grouped eval loader, cell weighting, and config/preset/target surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from decision_engine.benchmark.metrics import (
    bootstrap_ci,
    compute_classification_and_calibration,
    compute_latency_percentiles,
    compute_permutation_drift,
)
from decision_engine.benchmark.scorecard import generate_json_report, generate_markdown_scorecard
from decision_engine.config import ExportConfig, TamevConfig, TrainingConfig
from decision_engine.config.presets import get_preset_config
from decision_engine.data.collator import TamevCollator
from decision_engine.data.dataset import KGroupedBatchSampler, TamevDataset
from decision_engine.data.schema import DecisionItem, OptionItem
from decision_engine.schema.decision import DecisionExample
from decision_engine.training.calibrator import TemperatureCalibrator
from decision_engine.training.finetune import finetune, plan_dataset_size
from decision_engine.training.hard_mining import mine_uncertain_examples
from decision_engine.training.losses import CompositeLoss, compute_brier_reward
from decision_engine.training.trainer import (
    TamevTrainer,
    aggregate_domain_metrics,
    compute_composite_score,
    cuda_amp_dtype,
)

# --------------------------------------------------------------------------------------
# Local stubs (no network, deterministic)
# --------------------------------------------------------------------------------------


class _StubTokenizer:
    """Whitespace tokenizer stand-in: no downloads, deterministic ids."""

    pad_token = "[PAD]"
    eos_token = "[EOS]"
    sep_token = "[SEP]"

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[EOS]": 1, "[SEP]": 2}

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        for text in texts:
            row = [self.vocab.setdefault(tok, len(self.vocab)) for tok in text.split()]
            ids.append(row[:max_length] if max_length else row)
        width = max((len(r) for r in ids), default=1) or 1
        padded = [r + [0] * (width - len(r)) for r in ids]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in ids]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)}


class _StubModel(nn.Module):
    """Parameter-free-forward stand-in; only needs parameters() for AdamW."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))


class _NoItemsDataset(Dataset):
    """Delegating wrapper with no `.items`, i.e. a dataset the sampler cannot group."""

    def __init__(self, inner: TamevDataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> DecisionItem:
        return self.inner[index]


def _make_items(count: int, num_options: int, target_text: str) -> list[DecisionItem]:
    items = []
    for i in range(count):
        options = [
            OptionItem(id=f"opt_{i}_{j}", text=target_text if j == 0 else f"distractor_{i}_{j}")
            for j in range(num_options)
        ]
        items.append(
            DecisionItem(
                id=f"item_{i}",
                state=f"state {i}",
                question=f"question {i}",
                options=options,
                label=options[0].id,
                task="general",
            )
        )
    return items


def _make_trainer(config: TamevConfig, tmp_path: Path, num_options: int = 12) -> TamevTrainer:
    config.training.device = "cpu"
    config.training.batch_size = 2
    config.training.output_dir = str(tmp_path / "run")
    items = _make_items(4, num_options, target_text="TARGET")
    dataset = TamevDataset(items)
    return TamevTrainer(
        model=_StubModel(),
        config=config,
        train_dataset=dataset,
        val_dataset=dataset,
        tokenizer=_StubTokenizer(),
    )


def _mixed_k_items(cardinalities: list[int]) -> list[DecisionItem]:
    """Interleaved option counts: K=77 rows sit next to K=2 rows, as in the real corpora."""
    items = []
    for i, k in enumerate(cardinalities):
        options = [OptionItem(id=f"o{i}_{j}", text=f"opt {i} {j}") for j in range(k)]
        items.append(
            DecisionItem(
                id=f"item_{i}",
                state=f"state {i}",
                question=f"question {i}",
                options=options,
                label=options[0].id,
                task="general",
            )
        )
    return items


def _make_mixed_k_trainer(tmp_path: Path, cardinalities: list[int], batch_size: int = 4):
    cfg = TamevConfig()
    cfg.training.device = "cpu"
    cfg.training.batch_size = batch_size
    cfg.training.output_dir = str(tmp_path / "run")
    dataset = TamevDataset(_mixed_k_items(cardinalities))
    return TamevTrainer(
        model=_StubModel(),
        config=cfg,
        train_dataset=dataset,
        val_dataset=dataset,
        tokenizer=_StubTokenizer(),
    )


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------


def test_composite_loss_terms_and_backward():
    loss_fn = CompositeLoss(
        ce_weight=1.0,
        brier_weight=0.5,
        margin_weight=0.5,
        kd_weight=0.5,
        margin_gamma=1.0,
        kd_temp=2.0,
    )
    logits = torch.randn(2, 4, requires_grad=True)
    targets = torch.tensor([0, 2], dtype=torch.long)
    teacher_probs = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]], dtype=torch.float32)

    out = loss_fn(logits, targets, teacher_probs, [4, 4])
    assert {"loss", "ce_loss", "brier_loss", "margin_loss", "kd_loss"} <= set(out)
    assert out["loss"].item() > 0.0

    out["loss"].backward()
    assert logits.grad is not None


def test_brier_reward_penalizes_overconfidence():
    r_overconfident_wrong = compute_brier_reward(pred_prob=0.99, is_correct=False)
    r_uncertain_wrong = compute_brier_reward(pred_prob=0.51, is_correct=False)
    r_confident_correct = compute_brier_reward(pred_prob=0.99, is_correct=True)

    assert r_confident_correct > r_uncertain_wrong
    assert r_uncertain_wrong > r_overconfident_wrong


def test_hard_example_mining():
    def _example(idx: str, probs: dict[str, float]) -> DecisionExample:
        return DecisionExample.from_dict(
            {
                "id": idx,
                "state": "State",
                "question": "Q",
                "decision_type": "choice",
                "options": [{"id": "A", "text": "A"}, {"id": "B", "text": "B"}],
                "label": "A",
                "teacher_probs": probs,
            }
        )

    ex_ambiguous = _example("1", {"A": 0.52, "B": 0.48})
    ex_confident = _example("2", {"A": 0.95, "B": 0.05})

    hard_list = mine_uncertain_examples([ex_ambiguous, ex_confident], uncertainty_margin=0.2)
    assert [e.id for e in hard_list] == ["1"]


def test_ordinal_penalty_charges_distance_to_the_target():
    """`score` cells are ordered: uniform over 5 with target index 2 costs |i-2| mass = 1.2."""
    loss_fn = CompositeLoss(
        ce_weight=0.0,
        brier_weight=0.0,
        margin_weight=0.0,
        kd_weight=0.0,
        rl_weight=0.0,
        ordinal_weight=1.0,
    )
    logits = torch.zeros(1, 5)

    charged = loss_fn(logits, torch.tensor([2]), num_options=[5], ordinal=torch.tensor([True]))
    assert abs(charged["ordinal_loss"].item() - 1.2) < 1e-6
    assert abs(charged["loss"].item() - 1.2) < 1e-6

    free = loss_fn(logits, torch.tensor([2]), num_options=[5], ordinal=torch.tensor([False]))
    assert free["ordinal_loss"].item() == 0.0


def test_ordinal_term_is_inert_without_a_positive_weight():
    """`ordinal_weight: 0.0` must leave the total loss unchanged even when the batch marks ordered
    items -- otherwise every shipped config would train differently."""
    common = {
        "ce_weight": 1.0,
        "brier_weight": 0.5,
        "margin_weight": 0.4,
        "kd_weight": 0.0,
        "rl_weight": 0.05,
    }
    logits = torch.randn(3, 5, generator=torch.Generator().manual_seed(0))
    targets = torch.tensor([1, 3, 4])
    marked = torch.tensor([True, False, True])

    off = CompositeLoss(**common, ordinal_weight=0.0)
    out = off(logits, targets, num_options=[5, 5, 5], ordinal=marked)
    unmarked = off(logits, targets, num_options=[5, 5, 5], ordinal=None)

    assert out["ordinal_loss"].item() == 0.0
    assert torch.equal(out["loss"], unmarked["loss"])

    on = CompositeLoss(**common, ordinal_weight=0.3)
    none_marked = on(
        logits, targets, num_options=[5, 5, 5], ordinal=torch.zeros(3, dtype=torch.bool)
    )
    assert none_marked["ordinal_loss"].item() == 0.0
    assert torch.equal(none_marked["loss"], unmarked["loss"])


def test_teacher_probs_drive_a_nonzero_kd_loss():
    """Pins the wiring collator -> has_teacher -> CompositeLoss, and that an unlabelled batch keeps
    the term at exactly zero."""
    collator = TamevCollator(tokenizer=_StubTokenizer(), max_ctx_len=8, max_opt_len=4)
    items = _make_items(2, num_options=3, target_text="TARGET")
    for item in items:
        teacher = item.options[item.target_index].id
        item.teacher_probs = {o.id: (0.8 if o.id == teacher else 0.1) for o in item.options}

    labelled = collator(items)
    assert labelled["has_teacher"].all()

    loss_fn = CompositeLoss(
        ce_weight=1.0, brier_weight=0.0, margin_weight=0.0, kd_weight=1.0, rl_weight=0.0
    )
    logits = torch.zeros(2, 3)  # uniform student: any KD signal shows up as the KD term itself
    out = loss_fn(
        logits,
        labelled["labels"],
        labelled["teacher_probs"],
        labelled["num_options"],
        labelled["has_teacher"],
        ordinal=labelled["ordinal"],
    )
    assert out["kd_loss"].item() > 0.0

    unlabelled = collator(_make_items(2, num_options=3, target_text="TARGET"))
    assert not unlabelled["has_teacher"].any()
    out2 = loss_fn(
        logits,
        unlabelled["labels"],
        unlabelled["teacher_probs"],
        unlabelled["num_options"],
        unlabelled["has_teacher"],
    )
    assert out2["kd_loss"].item() == 0.0


# --------------------------------------------------------------------------------------
# Cell weights
# --------------------------------------------------------------------------------------


def test_cell_weights_default_is_bit_identical():
    """`cell_weights` absent/empty must leave every loss term bit-for-bit equal to the pre-knob path."""
    common = {
        "ce_weight": 1.0,
        "brier_weight": 0.5,
        "margin_weight": 0.4,
        "kd_weight": 0.3,
        "rl_weight": 0.05,
    }
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(4, 5, generator=g)
    targets = torch.tensor([0, 2, 4, 1])
    teacher_probs = torch.softmax(torch.randn(4, 5, generator=g), dim=-1)
    has_teacher = torch.tensor([True, False, True, False])
    num_options = [5, 5, 5, 5]
    cells = ["yelp/rating", "amazon/stars", "sst5/sentiment", "banking77/intent"]
    keys = ("loss", "ce_loss", "brier_loss", "margin_loss", "kd_loss", "rl_loss", "ordinal_loss")

    today = CompositeLoss(**common)(logits, targets, teacher_probs, num_options, has_teacher)

    for table in (None, {}):
        gated = CompositeLoss(**common, cell_weights=table)(
            logits, targets, teacher_probs, num_options, has_teacher, cells=cells
        )
        for key in keys:
            assert torch.equal(gated[key], today[key]), key


def test_cell_weights_upweight_the_named_cell_and_normalise():
    """A heavier weight on the error-heavy cell raises the loss; the table is renormalised to a
    batch mean of 1.0, so scaling the whole table by a constant is a no-op."""
    common = {
        "ce_weight": 1.0,
        "brier_weight": 0.0,
        "margin_weight": 0.0,
        "kd_weight": 0.0,
        "rl_weight": 0.0,
    }
    # item 0 correct (target 0), item 1 wrong (same logits, target 1)
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    targets = torch.tensor([0, 1])
    cells = ["sst5/sentiment", "yelp/rating"]

    plain = CompositeLoss(**common)(logits, targets, cells=cells)
    weighted = CompositeLoss(**common, cell_weights={"yelp/rating": 3.0})(
        logits, targets, cells=cells
    )
    assert weighted["loss"].item() > plain["loss"].item()

    uniform = CompositeLoss(**common, cell_weights={"sst5/sentiment": 2.0, "yelp/rating": 2.0})(
        logits, targets, cells=cells
    )
    assert torch.allclose(uniform["loss"], plain["loss"], atol=1e-7)

    full_a = CompositeLoss(**common, cell_weights={"sst5/sentiment": 1.0, "yelp/rating": 2.0})(
        logits, targets, cells=cells
    )
    full_b = CompositeLoss(**common, cell_weights={"sst5/sentiment": 2.0, "yelp/rating": 4.0})(
        logits, targets, cells=cells
    )
    assert torch.allclose(full_a["loss"], full_b["loss"], atol=1e-7)

    # The returned ce_loss is the renormalised weighted mean, not a raw sum.
    ce_item = -torch.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    w = torch.tensor([1.0, 3.0])
    w = w / w.mean()
    assert torch.equal(weighted["ce_loss"], (ce_item * w).mean())


def test_cell_weights_are_invariant_to_option_permutation():
    """Weights are per-item constants keyed by cell, independent of option order or K."""
    common = {
        "ce_weight": 1.0,
        "brier_weight": 0.5,
        "margin_weight": 0.4,
        "kd_weight": 0.0,
        "rl_weight": 0.0,
    }
    loss_fn = CompositeLoss(**common, cell_weights={"yelp/rating": 2.5})
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(3, 4, generator=g)
    targets = torch.tensor([1, 3, 0])
    cells = ["yelp/rating", "amazon/stars", "yelp/rating"]

    base = loss_fn(logits, targets, cells=cells)["loss"]
    perm = torch.tensor([2, 0, 3, 1])
    permuted = loss_fn(logits[:, perm], torch.argsort(perm)[targets], cells=cells)["loss"]

    assert (permuted - base).abs().item() <= 1e-8


def test_cell_weights_config_surface_collator_and_trainer_wiring(tmp_path: Path):
    """`training.cell_weights` survives a YAML round-trip, the collator emits the "<source>/<qid>"
    label the lookup keys on, and the trainer hands the table to the loss."""
    cfg = TamevConfig()
    cfg.training.cell_weights = {"yelp/rating": 2.0}
    path = tmp_path / "cell_weights.yaml"
    cfg.to_yaml(path)
    assert TamevConfig.from_yaml(path).training.cell_weights == {"yelp/rating": 2.0}

    items = _make_items(1, 3, target_text="TARGET")
    items[0].metadata = {"source": "yelp", "qid": "rating"}
    batch = TamevCollator(tokenizer=_StubTokenizer(), max_ctx_len=8, max_opt_len=4)(items)
    assert batch["cells"] == ["yelp/rating"]

    assert _make_trainer(cfg, tmp_path).loss_fn.cell_weights == {"yelp/rating": 2.0}


# --------------------------------------------------------------------------------------
# Collator
# --------------------------------------------------------------------------------------


def test_trainer_honors_dataset_max_options(tmp_path: Path):
    cfg = TamevConfig()
    cfg.dataset.max_options = 3
    trainer = _make_trainer(cfg, tmp_path)
    assert trainer.train_collator.max_options == 3
    assert trainer.train_collator([trainer.train_dataset[0]])["num_options"] == [3]

    assert _make_trainer(TamevConfig(), tmp_path).train_collator.max_options == 8


@pytest.mark.parametrize("target_pos", [0, 5, 11])
def test_collator_subsampling_preserves_target_option(target_pos):
    collator = TamevCollator(
        tokenizer=_StubTokenizer(), max_ctx_len=32, max_opt_len=8, max_options=4, seed=0
    )
    target_token_id = collator.tokenizer("TARGET")["input_ids"][0, 0].item()

    options = [OptionItem(id=f"o{j}", text=f"d{j}") for j in range(12)]
    options[target_pos] = OptionItem(id="target", text="TARGET")
    item = DecisionItem(
        id="x", state="s", question="q", options=options, label="target", task="general"
    )

    batch = collator([item])
    assert batch["num_options"] == [4]
    label = batch["labels"][0].item()
    assert batch["opt_input_ids"][0, label, 0].item() == target_token_id


def test_collator_marks_only_score_items_ordinal():
    collator = TamevCollator(tokenizer=_StubTokenizer(), max_ctx_len=8, max_opt_len=4)
    score = _make_items(1, num_options=5, target_text="Rating 2")[0]
    score.decision_type = "score"
    choice = _make_items(1, num_options=5, target_text="Rating 2")[0]

    assert collator([score, choice])["ordinal"].tolist() == [True, False]


# --------------------------------------------------------------------------------------
# Calibration + trainer
# --------------------------------------------------------------------------------------


def test_calibration_metrics():
    probs = np.array([[0.1, 0.8, 0.1], [0.1, 0.9, 0.0]], dtype=np.float32)
    targets = [1, 0]  # Ex 0 correct (pred 1), Ex 1 wrong (pred 1 != 0)

    metrics = compute_classification_and_calibration(list(probs), targets)
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["brier_score"] > 0.0
    assert metrics["nll"] > 0.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_temperature_calibrator():
    calibrator = TemperatureCalibrator()
    logits = [torch.tensor([8.0, 1.0]), torch.tensor([8.0, 1.0])]  # overconfident, 50% accurate
    targets = [0, 1]
    assert calibrator.calibrate(logits, targets) > 1.0


def test_resume_checkpoint_round_trips(tmp_path: Path):
    """A preempted run continues from <out_dir>/tamev_model_resume.pt; a torn file would be worse."""
    trainer = _make_trainer(TamevConfig(), tmp_path)
    path = trainer.output_dir / "tamev_model_resume.pt"

    trainer._save_resume(path, epoch=2, step=100, best_val_score=0.42)

    assert path.exists() and not path.with_suffix(".tmp").exists()
    assert trainer._load_resume(path) == (2, 100, 0.42)


def test_eval_loader_is_k_grouped_and_keeps_every_row(tmp_path: Path):
    """Val/test batches pad to the batch max_K, so they must be K-grouped (3.0x padded tokens), and
    the ragged tail must not be dropped."""
    cardinalities = [2, 77, 3, 14] * 16 + [5]
    trainer = _make_mixed_k_trainer(tmp_path, cardinalities, batch_size=4)
    sampler = trainer.val_loader.batch_sampler
    assert isinstance(sampler, KGroupedBatchSampler)
    assert sampler.shuffle is False, "eval order must stay deterministic"

    grouped = list(sampler)
    assert sorted(i for b in grouped for i in b) == list(range(len(cardinalities)))

    sequential = [
        list(range(start, min(start + 4, len(cardinalities))))
        for start in range(0, len(cardinalities), 4)
    ]
    padded = lambda batches: sum(max(cardinalities[i] for i in b) for b in batches)  # noqa: E731
    assert padded(grouped) * 2 < padded(sequential), (padded(grouped), padded(sequential))

    grouped_labels = [trainer.val_dataset[i].target_index for b in grouped for i in b]
    assert sorted(grouped_labels) == sorted(
        trainer.val_dataset[i].target_index for i in range(len(cardinalities))
    )


def test_eval_loader_falls_back_when_rows_are_not_groupable(tmp_path: Path):
    """A val dataset without `.items` (e.g. a streaming wrapper) keeps the plain loader."""
    cfg = TamevConfig()
    cfg.training.device = "cpu"
    cfg.training.batch_size = 2
    cfg.training.output_dir = str(tmp_path / "run")
    dataset = TamevDataset(_mixed_k_items([2, 3, 4]))
    trainer = TamevTrainer(
        model=_StubModel(),
        config=cfg,
        train_dataset=dataset,
        val_dataset=_NoItemsDataset(dataset),
        tokenizer=_StubTokenizer(),
    )
    assert not isinstance(trainer.val_loader.batch_sampler, KGroupedBatchSampler)
    assert trainer.val_loader.batch_size == 2


def test_seq_eval_loader_still_produces_identical_metrics(tmp_path: Path):
    """Grouping changes batch composition only: the collated tensors must cover the same rows."""
    cardinalities = [2, 5, 5, 3, 2, 4]
    trainer = _make_mixed_k_trainer(tmp_path, cardinalities, batch_size=2)
    sequential = DataLoader(
        trainer.val_dataset, batch_size=2, shuffle=False, collate_fn=trainer.eval_collator
    )

    def flatten(loader):
        rows = []
        for batch in loader:
            for i, k in enumerate(batch["num_options"]):
                rows.append((int(k), int(batch["labels"][i])))
        return sorted(rows)

    assert flatten(trainer.val_loader) == flatten(sequential)


def test_grad_scaler_is_disabled_off_the_fp16_path(tmp_path: Path):
    """Scaler must be inert for bf16/fp32/MPS/CPU; enabling it there would rescale valid grads."""
    assert _make_trainer(TamevConfig(), tmp_path).scaler.is_enabled() is False


@pytest.mark.parametrize(
    "sm,expected", [(7, torch.float16), (8, torch.bfloat16), (9, torch.bfloat16)]
)
def test_cuda_amp_dtype_follows_native_bf16_support(sm, expected):
    """sm_75 (T4) has no native bf16 kernels: fp16 + GradScaler there, bf16 from sm_80 up."""
    assert cuda_amp_dtype(sm) is expected


# --------------------------------------------------------------------------------------
# Early stopping + benchmark metrics
# --------------------------------------------------------------------------------------


def test_composite_metric_calculation():
    # Top1 = 0.70, Top3 = 0.90 -> 0.7*0.70 + 0.3*0.90 = 0.76
    assert abs(compute_composite_score(top1=0.70, top3=0.90) - 0.76) < 1e-5


def test_training_config_has_min_delta():
    cfg = TrainingConfig(early_stopping_min_delta=0.002, early_stopping_metric="composite")
    assert cfg.early_stopping_min_delta == 0.002
    assert cfg.early_stopping_metric == "composite"


def test_classification_and_calibration_metrics():
    probs = [np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.7, 0.2]), np.array([0.3, 0.3, 0.4])]
    labels = [0, 1, 2]  # all correct
    res = compute_classification_and_calibration(probs, labels)

    assert res["top1_accuracy"] == 1.0
    assert res["top3_accuracy"] == 1.0
    assert 0.0 <= res["ece"] <= 0.5
    assert 0.0 <= res["brier_score"] <= 1.0
    assert res["n"] == 3


def test_permutation_drift_metric():
    orig_p = np.array([0.7, 0.2, 0.1])
    perm_p = np.array([0.7, 0.2, 0.1])  # perfectly aligned
    drift, flip = compute_permutation_drift([orig_p], [perm_p])
    assert drift == 0.0
    assert flip == 0


def test_latency_percentiles():
    stats = compute_latency_percentiles([4.2, 4.5, 4.8, 5.0, 5.5, 6.0, 7.0, 10.0])
    assert {"p50", "p95", "mean", "qps"} <= set(stats)
    assert stats["p50"] <= stats["p95"]


def test_bootstrap_ci_is_bounded_and_deterministic():
    rng = np.random.default_rng(0)
    probs = [rng.dirichlet(np.ones(4)).astype(np.float32) for _ in range(120)]
    labels = [int(np.argmax(p)) if i % 3 else 0 for i, p in enumerate(probs)]

    lo, hi = bootstrap_ci(probs, labels, "ece", n_boot=100, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    assert bootstrap_ci(probs, labels, "ece", n_boot=100, seed=0) == (lo, hi)
    assert bootstrap_ci([], [], "ece", n_boot=10) == (0.0, 0.0)


def test_scorecard_and_json_report(tmp_path: Path):
    report_data = {
        "model_name": "tamev_test_model",
        "clean_metrics": {
            "top1_accuracy": 0.5034,
            "top3_accuracy": 0.8898,
            "ece": 0.0405,
            "brier_score": 0.5806,
            "n": 735,
        },
        "permutation_metrics": {"mean_max_drift": 0.00000000, "flip_rate": 0.0},
        "latency_metrics": {"p50": 4.8, "p95": 6.2, "mean": 5.1, "qps": 196},
        "task_breakdown": {
            "banking": {"top1_accuracy": 0.48, "ece": 0.038, "n": 100},
            "ecommerce": {"top1_accuracy": 0.55, "ece": 0.041, "n": 100},
        },
    }
    assert generate_json_report(report_data, tmp_path / "report.json").exists()

    md_path = generate_markdown_scorecard(report_data, tmp_path / "scorecard.md")
    assert md_path.exists()
    content = md_path.read_text()
    assert "Top-1 Accuracy" in content
    assert "50.34%" in content


# --------------------------------------------------------------------------------------
# Config / presets
# --------------------------------------------------------------------------------------


def test_default_config_instantiation():
    cfg = TamevConfig()
    assert cfg.model.backbone == "huawei-noah/TinyBERT_General_4L_312D"
    assert cfg.model.size_profile == "nano"
    assert cfg.model.temperature == 2.2
    assert cfg.teacher.model_path == "Qwen/Qwen3.5-4B"
    assert cfg.dataset.max_options == 8
    assert cfg.training.kd_temperature == 2.0
    assert cfg.target.snake_score == 500
    assert cfg.target.tetris_score == 2000


@pytest.mark.parametrize(
    "preset,size_profile,backbone_substr",
    [
        ("nano", "nano", "tinybert"),
        ("micro", "micro", "minilm"),
        ("small_modernbert", "small", "modernbert"),
        ("medium_qwen05b", "medium", "qwen"),
    ],
)
def test_size_profiles(preset, size_profile, backbone_substr):
    cfg = get_preset_config(preset)
    assert cfg.model.size_profile == size_profile
    # Case-insensitive: lock the *family*, not one vendor's casing (gte-modernbert, not answerdotai).
    assert backbone_substr in cfg.model.backbone.lower()


def test_yaml_roundtrip(tmp_path: Path):
    cfg = TamevConfig()
    cfg.model.backbone = "sentence-transformers/all-MiniLM-L6-v2"
    cfg.model.size_profile = "micro"
    cfg.training.learning_rate = 5e-5
    cfg.benchmark.suite_path = "data/custom_test.jsonl"
    cfg.target.snake_score = 600
    cfg.target.tetris_score = 2500

    yaml_file = tmp_path / "test_config.yaml"
    cfg.to_yaml(yaml_file)
    assert yaml_file.exists()

    loaded = TamevConfig.from_yaml(yaml_file)
    assert loaded.model.backbone == "sentence-transformers/all-MiniLM-L6-v2"
    assert loaded.model.size_profile == "micro"
    assert loaded.training.learning_rate == 5e-5
    assert loaded.benchmark.suite_path == "data/custom_test.jsonl"
    assert loaded.target.snake_score == 600
    assert loaded.target.tetris_score == 2500


def test_export_config_precisions():
    cfg = ExportConfig(precisions=["fp32", "fp16", "int8"], targets=["onnx", "pytorch"])
    assert "int8" in cfg.precisions
    assert "onnx" in cfg.targets


def test_target_config_fallback():
    # Target omitted -> defaults.
    cfg = TamevConfig.from_dict({})
    assert cfg.target.snake_score == 500
    assert cfg.target.tetris_score == 2000

    # Partial target dict -> other fields fall back.
    cfg_partial = TamevConfig.from_dict({"target": {"snake_score": 750}})
    assert cfg_partial.target.snake_score == 750
    assert cfg_partial.target.tetris_score == 2000


def test_build_config_from_args():
    import argparse

    from scripts.train import build_config_from_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--preset", default="nano")
    parser.add_argument("--model-type", default="encoder")
    parser.add_argument("--backbone", default="huawei-noah/TinyBERT_General_4L_312D")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--teacher-provider", default="mock")
    parser.add_argument("--teacher-model", default="mock-v1")
    parser.add_argument("--teacher-temp", type=float, default=2.5)
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs/test_run")
    parser.add_argument("--kd-weight", type=float, default=0.8)

    cfg = build_config_from_args(parser.parse_args([]))
    assert cfg.model.model_type == "encoder"
    assert cfg.model.backbone == "huawei-noah/TinyBERT_General_4L_312D"
    assert cfg.model.projection_dim == 128
    assert cfg.model.temperature == 1.8
    assert cfg.teacher.provider == "mock"
    assert cfg.teacher.temperature == 2.5
    assert cfg.training.epochs == 5
    assert cfg.training.batch_size == 16
    assert cfg.training.kd_weight == 0.8


# --------------------------------------------------------------------------------------
# Fine-tune pipeline
# --------------------------------------------------------------------------------------


def _write_decision_jsonl(path: Path, examples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def test_plan_dataset_size_balanced(tmp_path: Path):
    sample_file = tmp_path / "sample_decisions.jsonl"
    _write_decision_jsonl(
        sample_file,
        [
            {
                "id": f"rec_{i}",
                "state": f"Customer enquiry {i}",
                "question": "Which department?",
                "options": [
                    {"id": "billing", "text": "Billing department"},
                    {"id": "tech", "text": "Technical support"},
                    {"id": "sales", "text": "Sales department"},
                ],
                "target_index": i % 3,
                "decision_type": "choice",
            }
            for i in range(60)
        ],
    )

    plan = plan_dataset_size(sample_file)
    assert plan["total_samples"] == 60
    assert plan["cardinality"]["avg"] == 3.0
    assert "Domain Specialization" in plan["adequacy"]
    assert plan["recommended_tier"] == "nano"
    assert "cpu" in plan["estimated_time_sec"]


def test_finetune_one_command_pipeline(tmp_path: Path):
    sample_file = tmp_path / "micro_train.jsonl"
    _write_decision_jsonl(
        sample_file,
        [
            {
                "id": f"rec_{i}",
                "state": f"Triage incident {i}: database pool connection timeout.",
                "question": "Choose incident response action:",
                "options": [
                    {"id": "scale_db", "text": "Scale connection pool capacity"},
                    {"id": "restart_svc", "text": "Restart affected backend worker service"},
                ],
                "target_index": i % 2,
                "decision_type": "choice",
            }
            for i in range(12)
        ],
    )

    out_dir = tmp_path / "run_finetune"
    res = finetune(
        data=sample_file,
        tier="nano",
        epochs=1,
        batch_size=4,
        device="cpu",
        output_dir=out_dir,
        export_formats="pytorch_fp32,onnx_int8,coreml,mlx,mps",
    )

    assert res["tier"] == "nano"
    assert res["epochs_trained"] == 1
    assert "calibrated_temperature" in res
    exported = out_dir / "exported"
    assert (exported / "deployment_manifest.json").exists()
    assert (exported / "tamev_nano_fp32.pt").exists()
    assert (exported / "tamev_nano_mps_fp16.pt").exists()
    assert (exported / "tamev_nano_int8.onnx").exists()
    assert (exported / "tamev_nano.mlpackage").exists()
    assert (exported / "tamev_nano_mlx" / "weights.npz").exists()
    assert (exported / "tamev_nano_mlx" / "tamev_mlx_config.json").exists()
