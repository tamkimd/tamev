# decision_engine/config/base.py
"""
Centralized, Strongly-Typed Hierarchical Configuration System for TAMEV.
Supports YAML/JSON serialization, dataclass validation, and CLI overrides.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ModelConfig:
    """Configuration for Student / Root Decision Model."""

    name: str = "tamev_tinybert"
    size_profile: str = "nano"
    model_type: Literal["encoder", "causal", "bi_encoder"] = "encoder"
    backbone: str = "huawei-noah/TinyBERT_General_4L_312D"
    hidden_dim: int | None = None
    projection_dim: int = 64
    temperature: float = 2.2
    calibration_by_k: dict[str, float] | None = None
    freeze_embeddings: bool = False
    freeze_backbone: bool = False
    pooling: Literal["cls", "mean", "last"] = "cls"
    normalize: bool = False
    option_fusion: Literal["bilinear", "cross"] = "bilinear"

    def __init__(
        self,
        name: str = "tamev_tinybert",
        size_profile: str = "nano",
        model_type: Literal["encoder", "causal", "bi_encoder"] = "encoder",
        backbone: str = "huawei-noah/TinyBERT_General_4L_312D",
        hidden_dim: int | None = None,
        projection_dim: int = 64,
        temperature: float = 2.2,
        calibration_by_k: dict[str, float] | None = None,
        freeze_embeddings: bool = False,
        freeze_backbone: bool = False,
        pooling: Literal["cls", "mean", "last"] = "cls",
        normalize: bool = False,
        option_fusion: Literal["bilinear", "cross"] = "bilinear",
        size_class: str | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self.size_profile = size_class if size_class is not None else size_profile
        self.model_type = model_type
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.calibration_by_k = calibration_by_k
        self.freeze_embeddings = freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling
        self.normalize = normalize
        self.option_fusion = option_fusion
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def size_class(self) -> str:
        return self.size_profile

    @size_class.setter
    def size_class(self, val: str):
        self.size_profile = val


@dataclass
class TeacherConfig:
    """Configuration for High-Capacity Teacher Model & Distillation Strategy."""

    name: str = "qwen3.5-4b"
    provider: Literal["huggingface", "vllm", "api", "opencode", "mock", "agent"] = "huggingface"
    model_path: str = "Qwen/Qwen3.5-4B"
    endpoint_url: str | None = None
    device: str = "auto"
    temperature: float = 2.0
    batch_size: int = 8
    max_samples: int = 1500
    cache_dir: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    prompt_template: str | None = None
    domain: str = "general"

    def __post_init__(self):
        import os

        if self.provider == "agent":
            if self.name in ("qwen3.5-4b", "Qwen3.5-4B"):
                self.name = (
                    os.environ.get("AGENT_MODEL_AS_TEACHER")
                    or os.environ.get("AGENT_MODEL_AS_TECHER")
                    or os.environ.get("AGENT_MODEL")
                    or os.environ.get("TAMEV_AGENT_MODEL")
                    or os.environ.get("TAMEV_AGENT_NAME")
                    or "agent-teacher"
                )
            if self.model_path in ("Qwen/Qwen3.5-4B", "qwen3.5-4b"):
                self.model_path = self.name


@dataclass
class DatasetConfig:
    """Configuration for Canonical & Task Datasets."""

    name: str = "canonical"
    train_path: str = "data/processed/train_merged.jsonl"
    val_path: str = "data/processed/val.jsonl"
    test_path: str = "data/processed/test.jsonl"
    max_ctx_len: int = 128
    max_opt_len: int = 48
    max_options: int | None = 8
    eval_max_options: int | None = None
    # Opt-in offline-mined hard negatives for the training collator (None = random subsampling)
    hard_negatives_path: str | None = None
    format: Literal["jsonl", "typesafe_v7", "hf"] = "jsonl"


@dataclass
class TrainingConfig:
    """Configuration for Training, Distillation, and RLCD Calibration."""

    stage: Literal["sft", "distill", "rlcd", "distill_rl", "full"] = "distill_rl"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    kd_temperature: float = 2.0
    kd_weight: float = 1.0
    margin_gamma: float = 1.2
    margin_weight: float = 0.5
    brier_weight: float = 0.5
    kl_anchor_weight: float = 0.1
    # Charges expected |option_index - target| on `decision_type: score` items (ordered options).
    # 0.0 = off; the flat softmax every shipped tier was trained with.
    ordinal_weight: float = 0.0
    # Optional per-cell sample weights keyed "<source>/<qid>" (e.g. "yelp/rating"), to upweight
    # the cells carrying the most error. Absent/empty => off, and the loss is bit-identical to the
    # pre-knob path. Renormalised per batch to mean 1.0 so a single knob cannot move the LR.
    cell_weights: dict[str, float] | None = None
    # R-Drop (Liang et al. 2021, arXiv:2106.14448): one extra forward pass with a fresh dropout
    # mask, plus a symmetric KL between the two output distributions. 0.0 = off, one forward pass,
    # bit-identical to the pre-knob path. Both passes score the options independently, so option
    # permutation still leaves every logit unchanged (drift stays exactly 0).
    rdrop_weight: float = 0.0
    # Noise-robust symmetric cross-entropy (Wang et al. 2019, arXiv:1908.06112). The added term is
    # `sce_weight * RCE`, RCE = -sum_k p_k log t_k, with the truth `t` smoothed by `sce_alpha` (a
    # one-hot t makes log t = -inf off-target). `sce_weight` is SCE's beta -- the CE half is already
    # term 1 -- so the pair is SCE with alpha=1.0, beta=sce_weight. 0.0 = off, bit-identical.
    sce_weight: float = 0.0
    sce_alpha: float = 0.1
    device: str = "auto"
    output_dir: str = "runs/tamev_distill_rl"
    seed: int = 42
    early_stopping_patience: int = 3
    early_stopping_metric: str = "composite"
    early_stopping_min_delta: float = 0.001
    amp: bool = True
    gradient_accumulation_steps: int = 1
    # "k_grouped" sorts by option cardinality so the collator pads to the batch's own K instead of
    # the corpus max (3.76x fewer padded option slots per epoch at batch 8). "random" keeps the
    # legacy uniform shuffle for A/B.
    batch_sampler: Literal["random", "k_grouped"] = "k_grouped"
    # A run that checkpoints only at epoch end loses everything if it dies mid-run. `resume` writes
    # an optimizer-inclusive checkpoint every `resume_every` optimizer steps (0 = off) and continues
    # from it on relaunch.
    resume: bool = False
    resume_every: int = 400


@dataclass
class ExportConfig:
    """Configuration for Multi-Format & Multi-Size Mobile/Edge Export."""

    precisions: list[str] = field(default_factory=lambda: ["fp32", "fp16", "int8"])
    targets: list[str] = field(default_factory=lambda: ["onnx", "pytorch"])
    onnx_opset: int = 17
    output_dir: str = "models/exported"


@dataclass
class BenchmarkConfig:
    """Configuration for Automated Benchmarking Suite."""

    suite_path: str = "data/processed/test.jsonl"
    output_dir: str = "runs/benchmarks"
    batch_size: int = 32
    num_permutations: int = 4
    latency_warmup: int = 10
    latency_reps: int = 50
    remote_endpoint: str | None = None


@dataclass
class ServerConfig:
    """Configuration for TypeSafe-Compatible Edge Server."""

    host: str = "127.0.0.1"
    port: int = 8008
    checkpoint_path: str | None = None
    device: str = "cpu"


@dataclass
class TargetConfig:
    """Target scores and quality thresholds for benchmarks, test evaluations, and live games."""

    snake_score: int = 500
    tetris_score: int = 2000
    top1_accuracy: float = 0.46
    top3_accuracy: float = 0.75
    max_ece: float = 0.05
    max_brier: float | None = 0.65
    max_latency_ms: float | None = 10.0


@dataclass
class TamevConfig:
    """Master Configuration for TAMEV Engine."""

    model: ModelConfig = field(default_factory=ModelConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TamevConfig:
        target_data = data.get("target", {})
        if isinstance(target_data, dict):
            import inspect

            valid_keys = set(inspect.signature(TargetConfig).parameters.keys())
            filtered = {k: v for k, v in target_data.items() if k in valid_keys}
            target_obj = TargetConfig(**filtered)
        else:
            target_obj = TargetConfig()

        return cls(
            model=ModelConfig(**data.get("model", {})),
            teacher=TeacherConfig(**data.get("teacher", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            training=TrainingConfig(**data.get("training", {})),
            export=ExportConfig(**data.get("export", {})),
            benchmark=BenchmarkConfig(**data.get("benchmark", {})),
            server=ServerConfig(**data.get("server", {})),
            target=target_obj,
        )

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        if yaml is not None:
            with open(p, "w", encoding="utf-8") as f:
                yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        else:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TamevConfig:
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) if yaml is not None and p.suffix != ".json" else json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_preset(cls, name: str) -> TamevConfig:
        from decision_engine.config.presets import get_preset_config

        return get_preset_config(name)
