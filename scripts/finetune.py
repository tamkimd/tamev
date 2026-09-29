#!/usr/bin/env python3
"""
CLI entry point for 1-command TAMEV fine-tuning.
Run:
    python scripts/finetune.py --plan-size --data data/processed/val_balanced.jsonl
    python scripts/finetune.py --data data/processed/train.jsonl --tier nano --epochs 3
"""

from decision_engine.training.finetune import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
