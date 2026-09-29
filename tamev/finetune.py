"""
TAMEV High-Level Fine-Tuning Interface.
Exposes `finetune(...)` and `plan_dataset_size(...)` for 1-command fine-tuning.
"""

from __future__ import annotations

from decision_engine.training.finetune import finetune, main, plan_dataset_size

__all__ = ["finetune", "main", "plan_dataset_size"]

if __name__ == "__main__":
    main()
