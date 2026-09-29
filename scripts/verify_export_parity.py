#!/usr/bin/env python3
"""Cross-runtime parity check for a published TAMEV repository.

The model card claims every shipped runtime returns the same decision. That claim is only true if
the ONNX / INT8 / TorchScript / FP16-MPS files were exported from the *same* weights as
`model.safetensors`, which a stale file in the folder silently breaks. This scores a sample of
`data/processed/test.jsonl` through each runtime and reports argmax agreement against the
safetensors path and against the documented `predict_decision` helper.

Needs `onnxruntime` and a directory that holds an exported bundle (see `tamev export`).

  uv run python scripts/verify_export_parity.py models/exported/nano [--n 64] [--seed 0]
  TAMEV_GATE=my_corpus.jsonl uv run python scripts/verify_export_parity.py models/exported/nano
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

from decision_engine.data.dataset import TamevDataset

REPO = Path(__file__).resolve().parent.parent
# Defaults to the corpus that ships with the repository; point TAMEV_GATE at a larger one when you
# have it. The frozen gate is not distributed -- see the README's Data section.
GATE = REPO / os.environ.get("TAMEV_GATE", "data/processed/test.jsonl")


def check_gate_sha() -> str | None:
    """The registry's sha256 for this gate, when it has one, so a truncated or swapped file cannot be
    scored under the gate's own name (benchmark_audit_v1.md Q5)."""
    reg = REPO / "runs/research/gate_registry.json"
    if not reg.exists():
        return None
    row = (json.loads(reg.read_text()) or {}).get(GATE.name) or {}
    want = row.get("sha256")
    if not want:
        return None
    got = hashlib.sha256(GATE.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(
            f"{GATE.name}: on-disk sha256 {got}\n  != sha recorded in gate_registry.json {want}"
        )
    return got


def load_items(n: int, seed: int):
    # Through the dataset loader, so `noul` / `score` rows get the same synthesised option set the
    # audit scored (raw jsonl rows carry `options: null` for those).
    items = TamevDataset(str(GATE)).items
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=min(n, len(items)), replace=False))
    return [items[i] for i in idx]


def tensors_for(item, tokenizer, max_state_len: int, max_opt_len: int):
    state = item.state
    question = item.question or ""
    context = f"{state}\n{question}".strip() if question else state
    opts = [o.text for o in item.options]
    c = tokenizer(
        context, max_length=max_state_len, padding="longest", truncation=True, return_tensors="pt"
    )
    o = tokenizer(
        opts, max_length=max_opt_len, padding="longest", truncation=True, return_tensors="pt"
    )
    return (
        c["input_ids"],
        c["attention_mask"],
        o["input_ids"].unsqueeze(0),
        o["attention_mask"].unsqueeze(0),
    ), opts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tier_dir")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tier_dir = Path(args.tier_dir)
    torch.set_num_threads(1)
    model = AutoModel.from_pretrained(str(tier_dir), trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(str(tier_dir), trust_remote_code=True)
    cfg = json.loads((tier_dir / "config.json").read_text())

    # A bloated INT8 ships silently when the exporter skips weight de-aliasing: the alias-MatMuls
    # stay FP32, so the "INT8" file is mostly float. Measured ratio int8/fp32 on the
    # gate-relevant tiers: de-aliased 0.26x (nano 14.18/54.71 MiB, micro 22.36/86.53 MiB); bloated
    # 0.58x / 0.73x. Anything above 0.35x means the quantizer regressed, not that the tier is small.
    fp32_mb = (tier_dir / "model.onnx").stat().st_size / 2**20
    int8_mb = (tier_dir / "model_int8.onnx").stat().st_size / 2**20
    if int8_mb > 0.35 * fp32_mb:
        raise SystemExit(
            f"INT8 BLOAT on {tier_dir.name}: {int8_mb:.2f} MiB vs fp32 {fp32_mb:.2f} MiB "
            f"(ratio {int8_mb / fp32_mb:.2f} > 0.35) -- weight de-aliasing is off"
        )

    sessions = {
        "onnx_fp32": ort.InferenceSession(
            str(tier_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        ),
        "onnx_int8": ort.InferenceSession(
            str(tier_dir / "model_int8.onnx"), providers=["CPUExecutionProvider"]
        ),
    }
    ts_model = torch.jit.load(str(tier_dir / "model.torchscript")).eval()
    fp16_state = torch.load(tier_dir / "model_mps_fp16.pt", map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    fp16_model = AutoModel.from_pretrained(str(tier_dir), trust_remote_code=True)
    fp16_model.load_state_dict(fp16_state, strict=False)
    fp16_model = fp16_model.eval()

    items = load_items(args.n, args.seed)
    rows = []
    for item in items:
        (ci, cm, oi, om), opts = tensors_for(item, tok, cfg["max_state_len"], cfg["max_opt_len"])
        target = int(item.target_index) if item.target_index is not None else -1
        names = ["ctx_input_ids", "ctx_attention_mask", "opt_input_ids", "opt_attention_mask"]
        feed = dict(zip(names, [t.numpy() for t in (ci, cm, oi, om)]))

        with torch.no_grad():
            base = model(
                ctx_input_ids=ci, ctx_attention_mask=cm, opt_input_ids=oi, opt_attention_mask=om
            )
            base_logits = base["logits"][0].float().numpy()
            ts_logits = np.asarray(ts_model(ci, cm, oi, om)[0])[0]
            fp16_logits = (
                fp16_model(
                    ctx_input_ids=ci, ctx_attention_mask=cm, opt_input_ids=oi, opt_attention_mask=om
                )["logits"][0]
                .float()
                .numpy()
            )
            doc_idx = model.predict_decision(
                state=item.state, question=item.question or "", options=opts, tokenizer=tok
            )["best_index"]

        row = {
            "n_options": len(opts),
            "target": target,
            "doc_idx": int(doc_idx),
            "argmax": {"safetensors": int(base_logits.argmax())},
        }
        for name, sess in sessions.items():
            # `asarray` first: onnxruntime's stub types the output as ndarray-or-SparseTensor, and
            # the row below indexes it.
            logits = np.asarray(sess.run(["logits"], feed)[0])[0]
            row["argmax"][name] = int(logits.argmax())
            row[f"dmax_{name}"] = float(np.abs(logits.astype(np.float64) - base_logits).max())
        row["argmax"]["torchscript"] = int(ts_logits.argmax())
        row["dmax_torchscript"] = float(np.abs(ts_logits.astype(np.float64) - base_logits).max())
        row["argmax"]["mps_fp16"] = int(fp16_logits.argmax())
        row["dmax_mps_fp16"] = float(np.abs(fp16_logits.astype(np.float64) - base_logits).max())
        rows.append(row)

    runtimes = [k for k in rows[0]["argmax"] if k != "safetensors"]
    report = {
        "tier_dir": str(tier_dir),
        "n": len(rows),
        # `TAMEV_GATE` may point outside the repository; then the absolute path is the honest one.
        "gate": str(GATE.relative_to(REPO)) if GATE.is_relative_to(REPO) else str(GATE),
    }
    for r in runtimes:
        agree = np.mean([row["argmax"][r] == row["argmax"]["safetensors"] for row in rows])
        agree_doc = np.mean([row["argmax"][r] == row["doc_idx"] for row in rows])
        dmax = max(row[f"dmax_{r}"] for row in rows)
        key = f"vs_safetensors_{r}"
        report[key] = {
            "argmax_agreement": round(float(agree), 4),
            "argmax_agreement_vs_predict_decision": round(float(agree_doc), 4),
            "max_abs_logit_delta": round(float(dmax), 6),
        }
    report["safetensors_vs_predict_decision"] = {
        "argmax_agreement": round(
            float(np.mean([row["argmax"]["safetensors"] == row["doc_idx"] for row in rows])), 4
        )
    }

    print(json.dumps(report, indent=2))
    # ponytail: dynamic INT8 has no calibration step, so it is not bit-exact by construction; it
    # only has to stay inside the tolerance the card publishes. Static (QDQ) quantization with the
    # 2,000-row calib split is the upgrade path if that tolerance is ever too wide.
    bad = [
        r
        for r in runtimes
        if report[f"vs_safetensors_{r}"]["argmax_agreement"] < (0.95 if r == "onnx_int8" else 1.0)
        or report[f"vs_safetensors_{r}"]["argmax_agreement_vs_predict_decision"]
        < (0.95 if r == "onnx_int8" else 1.0)
    ]
    if bad:
        raise SystemExit(f"PARITY FAIL on {tier_dir.name}: {bad}")
    print(f"PARITY OK: {tier_dir.name} ({len(rows)} items x {len(runtimes)} runtimes)")


if __name__ == "__main__":
    main()
