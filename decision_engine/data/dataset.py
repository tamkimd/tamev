# decision_engine/data/dataset.py
"""
PyTorch Dataset for TAMEV Decision Engine.
Supports JSONL streaming, Kev v7 conversion, and on-the-fly permutation augmentation.
"""

import json
import math
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Sampler

from decision_engine.data.schema import DecisionItem
from decision_engine.data.typesafe_adapter import parse_raw_sample_to_items


class KGroupedBatchSampler(Sampler[list[int]]):
    """Batch indices by similar option cardinality instead of uniformly at random.

    ``TamevCollator`` pads every row in a batch to that batch's ``max_k = max(k_list)``, so the
    padded work per batch is ``batch_size * max_K`` (times the batch's longest option). A corpus
    with mean K 16.8 but 6,000 K=77 rows pads a random batch to a mean K of 63.1; measured at
    batch 8 that is 2.14M option slots per epoch versus 0.57M grouped -- 3.76x less work. Grouping
    also keeps same-K gradients together rather than mixing noul (K=2) rows with wide-K (K=77) rows.

    Rows are sorted by K and reshuffled inside a sliding ``window`` each epoch, so the pairing
    changes between epochs without widening the pad. ``shuffle=False`` is the deterministic
    evaluation order.

    ``keys`` optionally widens the sort to ``(K, option-length proxy)``, for corpora where option
    lengths vary inside one K group. ponytail: unmeasured at arm level -- it saves 0.2% of padded
    option tokens (option texts are near-uniform inside a cell: mean 7 tokens, batch max
    9.9), so nothing wires it up. Add the secondary key in the trainer when a corpus has genuinely
    heterogeneous option lengths and the padded-token measurement moves.
    """

    def __init__(
        self,
        cardinalities: list[int],
        batch_size: int,
        seed: int = 0,
        *,
        shuffle: bool = True,
        window: int = 4,
        drop_last: bool = False,
        keys: list[tuple[int, ...]] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if keys is not None and len(keys) != len(cardinalities):
            raise ValueError(
                f"keys has {len(keys)} entries, cardinalities has {len(cardinalities)}"
            )
        self.cardinalities = [int(k) for k in cardinalities]
        self.keys = [tuple(k) for k in keys] if keys is not None else None
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.window = max(1, window)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.cardinalities)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def _order_key(self, i: int) -> tuple[int, ...]:
        # K first, then the tiebreak; the tiebreak is only consulted inside a K group.
        return self.keys[i] if self.keys is not None else (self.cardinalities[i],)

    def __iter__(self):
        order = sorted(range(len(self.cardinalities)), key=self._order_key)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            for start in range(0, len(order), self.window):
                chunk = order[start : start + self.window]
                rng.shuffle(chunk)
                order[start : start + self.window] = chunk
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            random.Random(self.seed + self.epoch + 1).shuffle(batches)
        yield from batches


class TamevDataset(Dataset):
    """
    Unified dataset supporting canonical JSONL and Kev/Jev v7 formats.
    """

    def __init__(
        self,
        data_source: str | Path | list[dict[str, Any]] | list[DecisionItem],
        shuffle_options_prob: float = 0.0,
        max_options: int | None = None,
    ):
        self.shuffle_options_prob = shuffle_options_prob
        self.max_options = max_options
        self.items: list[DecisionItem] = []

        if isinstance(data_source, (str, Path)):
            p = Path(data_source)
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        parsed = parse_raw_sample_to_items(raw)
                        self.items.extend(parsed)
                    except Exception:
                        continue
        elif isinstance(data_source, list):
            for entry in data_source:
                if isinstance(entry, DecisionItem):
                    self.items.append(entry)
                elif isinstance(entry, dict):
                    self.items.extend(parse_raw_sample_to_items(entry))

        # Filter items with fewer than 2 options
        self.items = [item for item in self.items if len(item.options) >= 2]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DecisionItem:
        item = self.items[idx]

        # Apply option permutation augmentation if enabled
        if self.shuffle_options_prob > 0.0 and random.random() < self.shuffle_options_prob:
            perm_options = list(item.options)
            random.shuffle(perm_options)
            # Reconstruct item with shuffled options
            item = DecisionItem(
                id=item.id,
                state=item.state,
                question=item.question,
                options=perm_options,
                label=item.label,
                task=item.task,
                decision_type=item.decision_type,
                teacher_probs=item.teacher_probs,
                metadata=item.metadata,
            )

        return item
