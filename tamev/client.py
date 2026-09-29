"""
TAMEV High-Level Client.
Provides a default-configured client for the local TAMEV System One edge server.
Inherits 100% of methods and capabilities from the official TypeSafeClient.
"""

import os
from typing import Any

from typesafe_sdk import (
    Choice,
    ListModelsResponse,
    ModelMetadata,
    Noul,
    Score,
    SystemOneResponse,
    TypeSafeClient,
)

DEFAULT_BASE_URL = os.environ.get("TAMEV_BASE_URL", "http://127.0.0.1:8008")
DEFAULT_API_KEY = os.environ.get("TAMEV_API_KEY", "local-edge")


class Client(TypeSafeClient):
    """
    TAMEV Edge System One Client.

    Convenience wrapper around official `TypeSafeClient`, defaulting to:
    - base_url: http://127.0.0.1:8008 (or TAMEV_BASE_URL env var)
    - api_key: 'local-edge' (or TAMEV_API_KEY env var)
    - model: 'tamev-latest'

    Example:
    ```python
    from tamev import Client, Choice, Noul, Score

    with Client() as client:
        # 1. Choice question
        res = client.system_one(
            state="Current tetris board has deep wells on column 9",
            questions={
                "action": Choice(
                    instructions="Which placement strategy should be chosen?",
                    criteria={
                        "fill_well": "Place I-bar in column 9 to clear 4 lines",
                        "flatten": "Place flat piece across columns 0-3",
                    },
                )
            },
        )
        print("Decision:", res.choices["action"].choice)
        print("Confidence:", res.choices["action"].confidence)
    ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            base_url=base_url if base_url is not None else DEFAULT_BASE_URL,
            api_key=api_key if api_key is not None else DEFAULT_API_KEY,
            model=model if model is not None else "tamev-latest",
            **kwargs,
        )


__all__ = [
    "Choice",
    "Client",
    "ListModelsResponse",
    "ModelMetadata",
    "Noul",
    "Score",
    "SystemOneResponse",
    "TypeSafeClient",
]
