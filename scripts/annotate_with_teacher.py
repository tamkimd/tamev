#!/usr/bin/env python3
"""
TAMEV Universal Pluggable Teacher Annotation Pipeline.
Annotates dataset samples with soft probability distributions from any configured teacher:
- HuggingFace local models (e.g. Qwen/Qwen3.5-4B, SmolLM2, Llama-3.2)
- API endpoints (vLLM, Ollama, OpenAI-compatible)
- Disk-cached teachers
- Mock teachers for testing and CI
"""

import argparse
import asyncio
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

from decision_engine.config.base import TeacherConfig
from decision_engine.data.schema import build_context
from decision_engine.data.typesafe_adapter import parse_raw_sample_to_items
from decision_engine.teacher.agent_teacher import resolve_agent_identity
from decision_engine.teacher.base import TeacherFactory


def annotate_dataset(
    input_path: str,
    output_path: str,
    provider: str = "mock",
    model_path: str | None = None,
    temperature: float = 2.0,
    max_samples: int = 1500,
    device: str = "auto",
    api_key: str | None = None,
    endpoint_url: str | None = None,
    strict: bool = True,
    cache_dir: str | None = None,
    batch_size: int = 16,
    sources: list[str] | None = None,
    per_source: int | None = None,
    concurrency: int = 8,
    flush_every: int = 500,
) -> dict[str, Any]:
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Dynamically resolve agent model if provider is agent
    resolved_model_path = model_path
    if provider == "agent":
        if not resolved_model_path or resolved_model_path in ("Qwen/Qwen3.5-4B", "qwen3.5-4b"):
            resolved_model_path = resolve_agent_identity()
    elif provider == "opencode":
        resolved_model_path = resolved_model_path or "deepseek-v4.1-flash"
    elif not resolved_model_path:
        resolved_model_path = "Qwen/Qwen3.5-4B"

    from decision_engine.utils.console import print_banner, print_kv_list

    print_banner("🎓 TAMEV Universal Teacher Annotation Pipeline")
    print_kv_list(
        [
            ("Provider", str(provider)),
            ("Model", str(resolved_model_path)),
            ("Temperature", str(temperature)),
            ("Input", str(in_file)),
            ("Output", str(out_file)),
            ("Max Samples", str(max_samples)),
        ]
    )

    t_cfg = TeacherConfig(
        name=Path(resolved_model_path).name if "/" in resolved_model_path else resolved_model_path,
        provider=provider,
        model_path=resolved_model_path,
        temperature=temperature,
        device=device,
        api_key=api_key,
        endpoint_url=endpoint_url,
        cache_dir=cache_dir,
    )
    teacher = TeacherFactory.create(t_cfg)
    if hasattr(teacher, "strict"):
        teacher.strict = strict
    print(f"[TEACHER] Initialized '{teacher.model_name}' successfully.")

    samples: list[dict[str, Any]] = []
    with open(in_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    continue

    print(f"[DATA] Loaded {len(samples)} samples from {in_file}.")

    t0 = time.time()
    already = sum(1 for ex in samples if ex.get("teacher_probs"))
    budget = max(0, max_samples - already)

    # The corpus is cell-major -- every agnews row, then every yelp row, and so on -- so a bare
    # `--max-samples` cap annotates one or two sources and calls the result a sample: 1500 rows of
    # agnews, none of banking77. `--sources` and `--per-source` exist because *which cells* get the
    # budget is the annotation decision, and the cells worth annotating are the ones the student
    # fails (banking77 .1136, sst5 .4122, boolq .5408, amazon .5121 for the nano tier).
    seen: dict[str, int] = {}
    todo: list[tuple[dict[str, Any], list[Any], str]] = []
    for ex in samples:
        if len(todo) >= budget:
            break
        if ex.get("teacher_probs"):
            continue
        src = str(ex.get("source") or "")
        if sources and src not in sources:
            continue
        # The option list the *trainer* will build, via the trainer's own parser. A hand-rolled copy
        # here used to synthesise the content-free "No / False" templates and write them into the
        # row, which schema.py:93-108 documents as putting both encoder tiers at or below the
        # majority-class baseline on 3 of the 4 agnews binary cells -- and because it wrote them
        # into the row, the trainer then used them too. It also handed the teacher a different
        # question than the student is graded on, so the soft labels described the wrong task.
        items = parse_raw_sample_to_items(ex)
        if not items or len(items[0].options) < 2:
            continue
        item = items[0]
        if per_source is not None and seen.get(src, 0) >= per_source:
            continue
        seen[src] = seen.get(src, 0) + 1
        # The identical string the collator tokenizes (`build_context`), untruncated: `max_ctx_len`
        # in the collator is the only length policy, so the teacher must not be shown a shorter
        # prefix of `state` than the student is scored on. `APITeacher._payload` caps at 2000 chars.
        todo.append((ex, item.options, build_context(item.state, item.question)))

    print(
        f"[PLAN] {len(todo)} rows to annotate ({already} already done); per source: "
        f"{dict(sorted(collections.Counter(str(e.get('source')) for e, _, _ in todo).items()))}"
    )

    sem = asyncio.Semaphore(max(1, concurrency))
    done = failed = uniform = 0

    def checkpoint() -> None:
        with open(out_file, "w", encoding="utf-8") as fh:
            for ex in samples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    async def call(context: str, texts: list[str]) -> Any:
        if hasattr(teacher, "aget_soft_targets"):
            return await teacher.aget_soft_targets(context, texts, temperature=temperature)
        # A local provider (huggingface/cached/vllm) has no async path; run it off the loop so the
        # semaphore still bounds how many are in flight and one slow row stalls nobody.
        return await asyncio.to_thread(teacher.get_soft_targets, context, texts, temperature)

    async def annotate_one(ex: dict[str, Any], opts: list[Any], context: str) -> None:
        nonlocal done, failed, uniform
        async with sem:
            try:
                probs = await call(context, [str(o.text) for o in opts])
            except Exception as e:
                # Leave the row unannotated on purpose. The collator sets `has_teacher=False` for
                # it and the KD term skips it, which is right; substituting a distribution here
                # would teach the student that every option is equally good -- a wrong label, not
                # a missing one, and the wrong-label kind is the one that costs top-1.
                failed += 1
                print(
                    f"  [{ex.get('id')}] teacher failed: {type(e).__name__}: {e}", file=sys.stderr
                )
                return
        if max(probs) - min(probs) < 1e-9:
            # What a `--no-strict` teacher returns on failure, identical in shape to a genuine
            # "all options equally good" answer. Drop it rather than label with it.
            uniform += 1
            return
        ex["teacher_probs"] = {str(o.id): round(float(p), 4) for o, p in zip(opts, probs)}
        done += 1
        if done % 50 == 0:
            rate = done / max(time.time() - t0, 0.01)
            print(f"  Annotated {done}/{len(todo)} ({rate:.1f} rows/sec)...", flush=True)
        if flush_every and done % flush_every == 0:
            # An annotation pass is hours long and the output used to be written only after the last
            # row, so a crash at hour two threw away every row paid for. Rewrite the file instead --
            # ~28 MB for this corpus, every few hundred rows. It also makes a restart cheap: rows
            # that already carry `teacher_probs` are skipped below.
            checkpoint()
            print(f"  checkpoint written ({done} annotated)", flush=True)

    async def run_all() -> None:
        # return_exceptions so one unexpected row cannot discard the whole pass.
        await asyncio.gather(
            *(annotate_one(ex, opts, ctx) for ex, opts, ctx in todo), return_exceptions=True
        )

    asyncio.run(run_all())
    annotated_count = already + done

    with open(out_file, "w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total_time = time.time() - t0
    print_banner("✅ Annotation Complete!")
    print_kv_list(
        [
            ("Annotated", f"{done} new, {annotated_count} total in {total_time:.1f}s"),
            ("Failed / uniform (left unannotated)", f"{failed} / {uniform}"),
            ("Output File", str(out_file)),
        ]
    )

    return {
        "annotated_samples": done,
        "annotated_total": annotated_count,
        "failed": failed,
        "uniform_skipped": uniform,
        "total_samples": len(samples),
        "output_path": str(out_file),
        "duration_sec": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="TAMEV Pluggable Teacher Annotation CLI")
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "agent", "huggingface", "api", "opencode", "cached", "vllm"],
        help="Teacher provider",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model name or HuggingFace path (defaults to active agent model for agent provider)",
    )
    parser.add_argument(
        "--input", type=str, default="data/processed/train_merged.jsonl", help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train_annotated.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--temperature", type=float, default=2.0, help="Distillation softening temperature"
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to annotate")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device (auto, mps, cuda, cpu)"
    )
    # No --api-key flag on purpose: a key passed on the command line is visible to every process on
    # the machine (`ps`) and lands in shell history. The teacher reads TAMEV_TEACHER_API_KEY, then
    # OPENCODE_API_KEY, then OPENAI_API_KEY from the environment (decision_engine/teacher/api_teacher.py).
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="OpenAI-compatible chat/completions URL (opencode default: OpenCode Zen)",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="On API failure fall back to uniform instead of raising",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Cache directory for cached provider"
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated `source` values to annotate, e.g. banking77,sst5,boolq,amazon,imdb. "
        "The corpus is cell-major, so without this a `--max-samples` cap spends the whole budget on "
        "whichever source comes first in the file.",
    )
    parser.add_argument(
        "--per-source",
        type=int,
        default=None,
        help="Cap rows per source, so a budget spread over several cells is not eaten by the "
        "largest one (agnews alone is 12,182 of 39,639 train rows).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Rows in flight at once. Annotation is network-bound, so wall time falls roughly "
        "linearly until the gateway rate-limits -- watch the 'Failed' count in the summary.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=500,
        help="Rewrite the output file every N annotated rows so a crash mid-pass does not discard "
        "the rows already paid for. 0 disables. A rerun resumes: rows that already carry "
        "`teacher_probs` are skipped.",
    )

    args = parser.parse_args()

    annotate_dataset(
        input_path=args.input,
        output_path=args.output,
        provider=args.provider,
        model_path=args.model_path,
        temperature=args.temperature,
        max_samples=args.max_samples,
        device=args.device,
        api_key=None,  # env only; see the note above the parser arguments
        endpoint_url=args.endpoint_url,
        strict=args.strict,
        cache_dir=args.cache_dir,
        sources=args.sources.split(",") if args.sources else None,
        per_source=args.per_source,
        concurrency=args.concurrency,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
