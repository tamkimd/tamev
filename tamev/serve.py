"""
TAMEV CLI Server Entry Point.
Run:
    python -m tamev.serve --port 8008 --device mps
"""

from __future__ import annotations

import argparse

from decision_engine.server.typesafe_server import run_server


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TAMEV Edge System One Decision Server")
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    run_server(
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        device=args.device,
        model_type=args.model_type,
        backbone=args.backbone,
        config=args.config,
    )


if __name__ == "__main__":
    main()
