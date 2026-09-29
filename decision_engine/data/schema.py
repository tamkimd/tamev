# decision_engine/data/schema.py
"""
Standardized Schema for Decision Examples in TAMEV.
Provides strongly typed dataclasses for decision tasks across all training stages.
"""

from dataclasses import dataclass, field
from typing import Any


def build_context(state: str, question: str = "") -> str:
    """The one context string -- training and serving must build it identically.

    Both serving paths (`HFDecisionModel.predict_decision` and the local engine) build
    `f"{state}\n{question}".strip()`, and the exported hf template inlines that same
    expression. Training used to disagree -- the collator built
    `f"{state[:200]} {sep} {question}"`, so every arm trained on roughly the first 50 tokens of
    state and was then scored on up to `max_ctx_len` tokens of it. Keep this expression
    character-for-character equal to `decision_engine/export/hf_templates/modeling_tamev.py`.
    """
    return f"{state}\n{question}".strip() if question else str(state)


@dataclass
class OptionItem:
    """A candidate option for decision selection."""

    id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptionItem":
        return cls(id=str(data["id"]), text=str(data["text"]))


@dataclass
class DecisionItem:
    """A single canonical decision task item."""

    id: str
    state: str
    question: str
    options: list[OptionItem]
    label: str  # id of the correct option
    task: str = "general"
    decision_type: str = "choice"
    teacher_probs: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_index(self) -> int:
        """Returns the 0-indexed position of the label in options."""
        target_str = str(self.label).strip().lower()
        for idx, opt in enumerate(self.options):
            if str(opt.id).strip().lower() == target_str:
                return idx
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "question": self.question,
            "options": [o.to_dict() for o in self.options],
            "label": self.label,
            "task": self.task,
            "decision_type": self.decision_type,
            "teacher_probs": self.teacher_probs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionItem":
        qtype = str(data.get("decision_type", "choice"))
        raw_opts = data.get("options")
        label_val = data.get("label", "")

        # `not raw_opts` gates BOTH synthetic-option paths, so a row that carries its own options
        # always uses them -- including `score` rows, which previously ignored them outright. The
        # synthetic `Rating i` template is content-free and says "Rating" even for the sentiment
        # cell; an explicit ordinal option list (e.g. "1 star - terrible") is what the pointer head
        # can match against the state. Absent options, the template is still what every shipped
        # manifest gets, so those files stay byte-identical and comparable.
        if not raw_opts and (qtype == "score" or data.get("scale") is not None):
            scale = data.get("scale") or {"min": 0, "max": 4}
            min_v = int(scale.get("min", 0))
            max_v = int(scale.get("max", 4))
            opts = [OptionItem(id=str(i), text=f"Rating {i}") for i in range(min_v, max_v + 1)]
            label_str = str(label_val if label_val is not None else min_v)
        elif not raw_opts:
            # Label-description training (arXiv 2305.02239): the content-free templates
            # ("No / False", "Yes / True") left the whole decision to the question string and put
            # both encoder tiers at or below the majority-class baseline on 3 of the 4 agnews
            # binary cells -- 3,198 items, 16.0% of the acceptance set. Carry the question into
            # both options so the head has lexical evidence to match against the state.
            # `not raw_opts` is the gate, exactly as above: a row that carries its own options --
            # noul included -- always uses them, so an instrument can express a noul option text
            # instead of having the synthesis silently overwrite it.
            q = str(data.get("question", "") or "").strip()
            opts = [
                OptionItem(id="false", text=f"no: {q}" if q else "No / False"),
                OptionItem(id="true", text=f"yes: {q}" if q else "Yes / True"),
            ]
            is_true = label_val in (True, "true", "yes", 1, "True")
            label_str = "true" if is_true else "false"
        else:
            opts = []
            for o in raw_opts:
                if isinstance(o, OptionItem):
                    opts.append(o)
                elif isinstance(o, dict):
                    opts.append(OptionItem.from_dict(o))
                elif isinstance(o, (list, tuple)) and len(o) >= 2:
                    opts.append(OptionItem(id=str(o[0]), text=str(o[1])))
                else:
                    opts.append(OptionItem(id=str(o), text=str(o)))
            if (label_val is None or str(label_val) == "") and "target_index" in data:
                try:
                    t_idx = int(data["target_index"])
                    label_str = opts[t_idx].id if 0 <= t_idx < len(opts) else str(t_idx)
                except Exception:
                    label_str = str(data.get("target_index", ""))
            else:
                label_str = str(label_val)

        task_val = (
            data.get("task")
            or data.get("metadata", {}).get("source")
            or (
                str(data.get("id", "")).split("/")[0]
                if "/" in str(data.get("id", ""))
                else "general"
            )
        )

        # Provenance has to survive the parse. The training corpora carry `source` and `qid` at the
        # top level and no `metadata` key at all, so `metadata` used to come out empty for every
        # row and `collator._cell_label` fell back to the split name. Everything keyed by
        # "<source>/<qid>" (the collator's `cells`, `training.cell_weights`, the audit's per-cell
        # breakdown) matched nothing and silently no-opped. An explicit `metadata` key still wins.
        metadata = {k: data[k] for k in ("source", "qid") if data.get(k) is not None}
        metadata.update(data.get("metadata") or {})

        return cls(
            id=str(data.get("id", "")),
            state=str(data.get("state", "")),
            question=str(data.get("question", "")),
            options=opts,
            label=label_str,
            task=str(task_val),
            decision_type=qtype,
            teacher_probs=data.get("teacher_probs"),
            metadata=metadata,
        )
