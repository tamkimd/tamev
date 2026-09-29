# decision_engine/data/kev_adapter.py
import json
from pathlib import Path
from typing import Any

from decision_engine.schema.decision import DecisionExample, DecisionType, OptionItem, ScoreScale


def convert_kev_record_to_canonical(
    record: dict[str, Any], record_id_prefix: str = "kev"
) -> list[DecisionExample]:
    """
    Converts a single Kev suite JSONL record (which can contain multiple questions)
    into a list of canonical DecisionExample objects with question isolation.
    """
    state_raw = record.get("state", "")
    if isinstance(state_raw, (dict, list)):
        state_text = json.dumps(state_raw, ensure_ascii=False)
    else:
        state_text = str(state_raw)

    meta = record.get("_meta", {})
    source_name = meta.get("source", "kev")
    rec_id = meta.get("id", f"{record_id_prefix}_{hash(state_text) % 1000000:06d}")

    examples = []
    questions_dict = record.get("questions", {})

    for qid, qdata in questions_dict.items():
        qtype = qdata.get("type", "choice")
        instr_raw = qdata.get("instructions", "")
        if isinstance(instr_raw, dict):
            question_text = instr_raw.get("question", str(instr_raw))
        else:
            question_text = str(instr_raw)

        raw_label = qdata.get("label")
        if raw_label is None:
            continue

        item_id = f"{rec_id}_{qid}"

        if qtype == "choice":
            criteria = qdata.get("criteria", {})
            if not criteria or len(criteria) < 2:
                continue

            options = []
            for opt_key, opt_desc in criteria.items():
                desc_text = str(opt_desc) if opt_desc is not None else opt_key.replace("_", " ")
                options.append(OptionItem(id=str(opt_key), text=desc_text))

            label_str = str(raw_label)
            # Verify label is among options
            if label_str not in {opt.id for opt in options}:
                continue

            ex = DecisionExample(
                id=item_id,
                state=state_text,
                question=question_text,
                decision_type=DecisionType.CHOICE,
                label=label_str,
                options=options,
                metadata={"source": source_name, "qid": qid, "num_options": len(options)},
            )
            examples.append(ex)

        elif qtype == "noul":
            ex = DecisionExample(
                id=item_id,
                state=state_text,
                question=question_text,
                decision_type=DecisionType.NOUL,
                label=bool(raw_label),
                metadata={"source": source_name, "qid": qid},
            )
            examples.append(ex)

        elif qtype == "score":
            criteria = qdata.get("criteria", [])
            max_val = len(criteria) - 1 if isinstance(criteria, list) and criteria else 5
            try:
                label_int = round(float(raw_label))
            except (ValueError, TypeError):
                continue

            ex = DecisionExample(
                id=item_id,
                state=state_text,
                question=question_text,
                decision_type=DecisionType.SCORE,
                scale=ScoreScale(min=0, max=max_val),
                label=label_int,
                metadata={"source": source_name, "qid": qid},
            )
            examples.append(ex)

    return examples


def convert_kev_file(input_jsonl: Path, output_jsonl: Path) -> int:
    """Converts a Kev JSONL file to canonical DecisionExample JSONL format."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        open(input_jsonl, encoding="utf-8") as f_in,
        open(output_jsonl, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            canonical_exs = convert_kev_record_to_canonical(rec)
            for ex in canonical_exs:
                f_out.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
                count += 1
    return count
