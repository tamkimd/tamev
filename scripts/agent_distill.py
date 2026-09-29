#!/usr/bin/env python3
"""
TAMEV Agentic Distillation Engine.

Guides autonomous AI coding agents (Claude Code, Cursor, Devin, ChatGPT, Codex, Windsurf, Aider) into the distillation pipeline using
structured prompt templates, domain rubrics, and response parsing.

Usage:
  # Mode 1: Directly annotate dataset with prompt-guided AgentTeacher
  python scripts/agent_distill.py --mode direct --input data/processed/val.jsonl --output data/processed/agent_distilled.jsonl --max-samples 50

  # Mode 2: Export prompt batch for AI Agent / Subagent review
  python scripts/agent_distill.py --mode export_prompts --input data/processed/val.jsonl --output runs/agent_prompts.json --max-samples 20

  # Mode 3: Ingest completed agent prompt responses into teacher cache
  python scripts/agent_distill.py --mode ingest --input runs/agent_responses.json --output data/processed/train_agent_annotated.jsonl
"""

import argparse
import json
from pathlib import Path

from decision_engine.teacher.agent_teacher import (
    AgentTeacher,
)


def export_agent_prompts(
    input_path: str,
    output_path: str,
    domain: str = "general",
    max_samples: int = 25,
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    teacher = AgentTeacher(domain=domain)
    prompts = []

    with open(in_file, encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            opts = item.get("options")
            if not opts or len(opts) < 2:
                continue

            state = str(item.get("state", ""))
            question = str(item.get("question", ""))
            ctx = f"{state}\n{question}".strip()
            opt_texts = [str(o.get("text", o.get("id", ""))) for o in opts]

            p_data = teacher.build_prompt(ctx, opt_texts, domain=domain)
            prompts.append(
                {
                    "id": item.get("id", f"sample_{len(prompts)}"),
                    "context": ctx,
                    "options": opt_texts,
                    "option_ids": [str(o.get("id", i)) for i, o in enumerate(opts)],
                    "system_prompt": p_data["system_prompt"],
                    "user_prompt": p_data["user_prompt"],
                }
            )

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)

    print(f"✅ Exported {len(prompts)} structured agent prompts to: {out_file}")


def ingest_agent_responses(
    responses_path: str,
    dataset_path: str,
    output_path: str,
    cache_path: str = "runs/agent_teacher_cache.json",
) -> None:
    resp_file = Path(responses_path)
    ds_file = Path(dataset_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    teacher = AgentTeacher(cache_path=cache_path)

    with open(resp_file, encoding="utf-8") as f:
        responses = json.load(f)

    resp_map = {}
    for entry in responses:
        item_id = entry.get("id")
        ctx = entry.get("context", "")
        options = entry.get("options", [])
        raw_resp = entry.get("response") or entry.get("probabilities")

        if isinstance(raw_resp, dict):
            probs = [float(raw_resp.get(opt, 1.0 / len(options))) for opt in options]
            parsed = {"probabilities": probs, "rationale": entry.get("rationale", "")}
        elif isinstance(raw_resp, str):
            parsed = teacher.parse_agent_response(raw_resp, options)
        else:
            continue

        teacher.register_manual_agent_decision(
            ctx,
            options,
            parsed["probabilities"].tolist()
            if hasattr(parsed["probabilities"], "tolist")
            else parsed["probabilities"],
            rationale=parsed.get("rationale", ""),
        )
        if item_id:
            resp_map[item_id] = parsed["probabilities"]

    print(f"📥 Successfully ingested {len(resp_map)} agent reasoning responses into teacher cache.")

    if ds_file.exists() and ds_file.is_file() and ds_file.suffix == ".jsonl":
        updated_lines = []
        annotated_ids = set()
        with open(ds_file, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                item = json.loads(line_str)
                item_id = item.get("id")
                if item_id in resp_map:
                    probs = resp_map[item_id]
                    opts = item.get("options", [])
                    item["teacher_probs"] = {
                        str(opts[i].get("id", i)): round(float(probs[i]), 4)
                        for i in range(len(opts))
                    }
                    item["teacher_model"] = teacher.model_name
                    annotated_ids.add(item_id)
                updated_lines.append(item)

        # For any items in resp_map not yet in the dataset, source base records from val.jsonl or train_merged.jsonl
        missing_ids = set(resp_map.keys()) - annotated_ids
        if missing_ids:
            for search_path in ["data/processed/val.jsonl", "data/processed/train_merged.jsonl"]:
                sp = Path(search_path)
                if sp.exists() and missing_ids:
                    with open(sp, encoding="utf-8") as f:
                        for line in f:
                            if not missing_ids:
                                break
                            line_str = line.strip()
                            if not line_str:
                                continue
                            item = json.loads(line_str)
                            item_id = item.get("id")
                            if item_id in missing_ids:
                                probs = resp_map[item_id]
                                opts = item.get("options", [])
                                item["teacher_probs"] = {
                                    str(opts[i].get("id", i)): round(float(probs[i]), 4)
                                    for i in range(len(opts))
                                }
                                item["teacher_model"] = teacher.model_name
                                updated_lines.append(item)
                                annotated_ids.add(item_id)
                                missing_ids.remove(item_id)

        with open(out_file, "w", encoding="utf-8") as f:
            for item in updated_lines:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"📝 Updated dataset {out_file} with {len(annotated_ids)} teacher-annotated samples.")


def direct_agent_annotate(
    input_path: str,
    output_path: str,
    domain: str = "general",
    temperature: float = 2.0,
    max_samples: int = 100,
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    teacher = AgentTeacher(domain=domain, temperature=temperature)
    annotated = []
    count = 0

    with open(in_file, encoding="utf-8") as f:
        for line in f:
            if count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            opts = item.get("options")
            if not opts or len(opts) < 2:
                continue

            state = str(item.get("state", ""))
            question = str(item.get("question", ""))
            ctx = f"{state}\n{question}".strip()
            opt_texts = [str(o.get("text", o.get("id", ""))) for o in opts]

            probs = teacher.get_soft_targets(ctx, opt_texts, temperature=temperature)
            item["teacher_probs"] = {
                str(opts[i].get("id", i)): round(float(probs[i]), 4) for i in range(len(opts))
            }
            item["teacher_model"] = teacher.model_name
            annotated.append(item)
            count += 1

    with open(out_file, "w", encoding="utf-8") as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"🎉 Directly annotated {len(annotated)} samples using prompt-guided AgentTeacher into: {out_file}"
    )


def main():
    parser = argparse.ArgumentParser(description="TAMEV Agent Distillation Prompt Engine")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "export_prompts", "ingest"],
        default="direct",
        help="Operation mode",
    )
    parser.add_argument(
        "--input", type=str, default="data/processed/train_merged.jsonl", help="Input dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset path to annotate with responses"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/agent_distilled.jsonl", help="Output path"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="general",
        choices=["general", "sre", "game_ai", "fintech"],
        help="Domain rubric",
    )
    parser.add_argument("--temperature", type=float, default=2.0, help="Softening temperature")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum samples to process")

    args = parser.parse_args()

    if args.mode == "export_prompts":
        export_agent_prompts(
            args.input, args.output, domain=args.domain, max_samples=args.max_samples
        )
    elif args.mode == "direct":
        direct_agent_annotate(
            args.input,
            args.output,
            domain=args.domain,
            temperature=args.temperature,
            max_samples=args.max_samples,
        )
    elif args.mode == "ingest":
        ingest_agent_responses(args.input, args.dataset or args.input, args.output)


if __name__ == "__main__":
    main()
