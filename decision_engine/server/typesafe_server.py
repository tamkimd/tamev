# decision_engine/server/typesafe_server.py
"""
TAMEV System One TypeSafe-Compatible Server.

Provides a 100% TypeSafe SDK and Kev-compatible System One API:
- POST /v1/systemone (supports Choice, Noul, Score questions)
- GET  /v1/models (returns ListModelsResponse metadata)
- POST /v1/systemone/permute (evaluates option order invariance)
- GET  /health (health check and latency statistics)

Enforces exact permutation invariance (0.0000 drift) and sub-10ms latency on Apple Silicon MPS & CPU.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import uuid

from aiohttp import web

from decision_engine.server.engine import TamevEngine
from decision_engine.server.handlers import (
    choice_confidence,
    option_text,
    render,
    round_prob,
    score_confidence,
)

logger = logging.getLogger("tamev.server")

# =====================================================================
# Model Catalog & Metadata
# =====================================================================

MODEL_CATALOG = [
    {
        "name": "tamev-latest",
        "description": "TAMEV Edge System One Decision Engine (TinyBERT-4L-312D Quantized INT8 / Distill RL)",
        "release_date": "2026-09-30",
    },
    {
        "name": "jev-latest",
        "description": "TAMEV Drop-In Replacement for Jev System One",
        "release_date": "2026-09-30",
    },
    {
        "name": "tamev-onnx-int8",
        "description": "TAMEV ONNX INT8 Quantized Edge Decision Model (31.3MB)",
        "release_date": "2026-09-30",
    },
]


def create_app(engine: TamevEngine | None = None) -> web.Application:
    """Creates aiohttp Application for TAMEV System One."""
    if engine is None:
        engine = TamevEngine()

    app = web.Application()

    # --- Middlewares ---
    @web.middleware
    async def cors_and_typesafe_middleware(request: web.Request, handler) -> web.Response:
        req_id = request.headers.get("x-typesafe-request-id") or f"req_{uuid.uuid4().hex[:16]}"

        if request.method == "OPTIONS":
            response = web.Response(status=204)
        else:
            # Optional Bearer Token Authentication
            api_key = os.environ.get("TAMEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
            if api_key and request.path.startswith("/v1"):
                auth_hdr = request.headers.get("Authorization", "")
                if not hmac.compare_digest(auth_hdr, f"Bearer {api_key}"):
                    return web.json_response(
                        {
                            "error": "Unauthorized",
                            "detail": "Missing or invalid Authorization Bearer key.",
                        },
                        status=401,
                        headers={
                            "x-typesafe-request-id": req_id,
                            "Access-Control-Allow-Origin": "*",
                        },
                    )
            try:
                response = await handler(request)
            except web.HTTPException as ex:
                response = ex
            except Exception as e:
                logger.exception("Error processing request: %s", e)
                response = web.json_response(
                    {"error": "InternalServerError", "detail": str(e)}, status=500
                )

        # TypeSafe & CORS Headers
        response.headers["x-typesafe-request-id"] = req_id
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "x-typesafe-request-id"
        return response

    app.middlewares.append(cors_and_typesafe_middleware)

    # --- Handlers ---
    async def handle_system_one(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": "BadRequest", "detail": "Malformed JSON body"}, status=400
            )

        state = body.get("state")
        if state is None:
            return web.json_response(
                {"error": "BadRequest", "detail": "Field 'state' is required."}, status=400
            )

        questions = body.get("questions")
        if not questions or not isinstance(questions, dict):
            return web.json_response(
                {"error": "BadRequest", "detail": "Field 'questions' must be a non-empty mapping."},
                status=400,
            )

        model_name = body.get("model", "tamev-latest")
        result = engine.predict_batch(state, questions, model_name=model_name)
        return web.json_response(result)

    async def handle_models(request: web.Request) -> web.Response:
        return web.json_response({"models": MODEL_CATALOG})

    async def handle_permute(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": "BadRequest", "detail": "Malformed JSON body"}, status=400
            )

        state = body.get("state")
        question = body.get("question")
        if not question or question.get("type") != "choice":
            return web.json_response(
                {"error": "BadRequest", "detail": "Permute test requires a 'choice' question."},
                status=400,
            )

        result = engine.check_permutation(state, question)
        return web.json_response(result)

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "service": "tamev",
                "model": "tamev-latest",
                "device": str(engine.device),
                "target_latency_ms": 15.0,
                "cache": engine.cache_stats,
            }
        )

    # --- Routes ---
    app.router.add_post("/v1/systemone", handle_system_one)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/systemone/permute", handle_permute)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/", handle_health)

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8008,
    checkpoint: str | None = None,
    device: str | None = None,
    model_type: str | None = None,
    backbone: str | None = None,
    config: str | None = None,
) -> None:
    """Starts the TAMEV TypeSafe-compatible server."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    engine = TamevEngine(
        checkpoint_path=checkpoint,
        device=device,
        model_type=model_type,
        backbone=backbone,
        config=config,
    )
    app = create_app(engine)
    logger.info(
        "Serving TAMEV System One (%s / %s) at http://%s:%d",
        engine.model_type,
        engine.backbone,
        host,
        port,
    )
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAMEV TypeSafe-Compatible System One Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8008, help="Port to listen on")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model weights checkpoint"
    )
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cpu, mps, cuda)")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Root model type (encoder, causal, bi_encoder, tinybert)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="HuggingFace model backbone repository or local path",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to Tamev YAML configuration file"
    )
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        device=args.device,
        model_type=args.model_type,
        backbone=args.backbone,
        config=args.config,
    )


__all__ = [
    "MODEL_CATALOG",
    "TamevEngine",
    "choice_confidence",
    "create_app",
    "option_text",
    "render",
    "round_prob",
    "run_server",
    "score_confidence",
]
