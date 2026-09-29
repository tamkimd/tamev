"""
TAMEV: Real-Time Edge AI Decision Engine.

Native TypeSafe SDK compatibility:
- Use official `TypeSafeClient` or convenient `tamev.Client`
- Formulate decisions with `Choice`, `Noul`, `Score`
- Encoder tiers (Nano/Micro/Small) score options permutation-invariantly; the in-context causal
  tiers (Medium/Large) are order-sensitive by construction
- Ultra-low latency (< 10 ms p50) for Nano/Micro on Apple Silicon and edge CPUs
"""

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    ListModelsResponse,
    ModelMetadata,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeClient,
    Usage,
)

from decision_engine.server.typesafe_server import TamevEngine, create_app, run_server

from .client import Client
from .finetune import finetune, plan_dataset_size
from .typesafe import TypeSafeDirectClient

__version__ = "0.0.1"

__all__ = [
    "Choice",
    "ChoiceAnswer",
    "Client",
    "ListModelsResponse",
    "ModelMetadata",
    "Noul",
    "NoulAnswer",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "TamevEngine",
    "TypeSafeClient",
    "TypeSafeDirectClient",
    "Usage",
    "__version__",
    "create_app",
    "finetune",
    "plan_dataset_size",
    "run_server",
]
