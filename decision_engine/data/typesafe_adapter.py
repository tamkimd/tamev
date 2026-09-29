# decision_engine/data/typesafe_adapter.py
"""
TypeSafe and Kev/Jev Canonical Data Adapter.
Converts heterogeneous dataset formats (Kev v7, TypeSafe API requests, raw JSONL)
into standardized DecisionItem instances.
"""

import uuid
from typing import Any

from decision_engine.data.schema import DecisionItem, OptionItem


def parse_raw_sample_to_items(raw_dict: dict[str, Any]) -> list[DecisionItem]:
    """
    Parses a single JSON line or dictionary into one or more DecisionItem instances.
    Supports:
    1. Canonical TAMEV JSONL format (with 'options' and 'label')
    2. Kev/Jev v7 format (with 'state' and 'questions' dict)
    """
    # 1. Canonical TAMEV or direct decision format
    if ("label" in raw_dict or "target_index" in raw_dict) and (
        "options" in raw_dict or "decision_type" in raw_dict or "question" in raw_dict
    ):
        return [DecisionItem.from_dict(raw_dict)]
    if "options" in raw_dict and ("state" in raw_dict or "question" in raw_dict):
        return [DecisionItem.from_dict(raw_dict)]

    # 2. Kev / Jev v7 System One Request format
    if "state" in raw_dict and "questions" in raw_dict:
        items = []
        state_str = str(raw_dict["state"])
        questions = raw_dict.get("questions", {})
        sample_id = raw_dict.get("id", str(uuid.uuid4())[:8])

        for q_id, q_data in questions.items():
            q_type = q_data.get("type", "choice")
            instructions = q_data.get("instructions", "")
            if isinstance(instructions, (dict, list)):
                instructions = str(instructions)

            label_val = q_data.get("label", None)
            options: list[OptionItem] = []

            if q_type == "choice":
                crit = q_data.get("criteria", {})
                for k, v in crit.items():
                    desc = f"{k}: {v}" if v else str(k)
                    options.append(OptionItem(id=str(k), text=desc))
                label_str = (
                    str(label_val) if label_val is not None else (options[0].id if options else "")
                )

            elif q_type == "noul":
                crit = q_data.get("criteria") or {}
                false_desc = crit.get("false") or crit.get(False) or "False / No"
                true_desc = crit.get("true") or crit.get(True) or "True / Yes"
                options = [
                    OptionItem(id="false", text=str(false_desc)),
                    OptionItem(id="true", text=str(true_desc)),
                ]
                is_true = label_val in (True, "true", "yes", 1)
                label_str = "true" if is_true else "false"

            elif q_type == "score":
                crit_list = q_data.get("criteria", [])
                for i, c in enumerate(crit_list):
                    options.append(OptionItem(id=str(i), text=str(c)))
                label_str = str(label_val) if label_val is not None else "0"

            else:
                continue

            items.append(
                DecisionItem(
                    id=f"{sample_id}_{q_id}",
                    state=state_str,
                    question=str(instructions),
                    options=options,
                    label=label_str,
                    decision_type=q_type,
                    task=str(q_data.get("src", "typesafe")),
                )
            )
        return items

    return []
