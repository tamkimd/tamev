---
name: tamev-finetune
description: Effortless 1-command fine-tuning and dataset planning for TAMEV decision models. Use when users want to fine-tune a model on custom decision datasets, plan required dataset size, evaluate accuracy tiers, calibrate temperature, or export to ONNX, Apple CoreML, and MLX.
---

# TAMEV Fine-Tuning Skill (`tamev-finetune`)

TAMEV provides an effortless 1-command fine-tuning experience. Unlike traditional LLMs that require expensive autoregressive token generation during training, TAMEV's decoupled bi-encoder pointer architecture enables complete model adaptation in **seconds to minutes** on standard CPUs, Apple Silicon Macs, or consumer GPUs.

---

## 🎯 When to Use This Skill

Activate this skill when:
- The user has custom domain decision logs (e.g. customer routing, triage, tool selection, content moderation, candidate ranking) and wants a dedicated, ultra-low latency model.
- The user asks how many samples are needed to train a decision model (`plan_dataset_size`).
- The user wants to optimize decision accuracy with proper scoring rules (Brier score + Margin ranking + KL anchor).
- The user wants calibrated probabilities ($ECE < 0.05$) matching true likelihood.
- The user wants automated multi-target export (PyTorch, ONNX INT8, Apple CoreML, Apple MLX).

---

## 📋 Data Schema

TAMEV accepts either **Canonical TAMEV JSONL** or **TypeSafe / Kev v7 format**:

### Format 1: Canonical TAMEV JSONL (Recommended)
```json
{
  "id": "triage_101",
  "state": "Kubernetes pod OOMKilled in payment-service namespace after traffic spike.",
  "question": "Select the immediate remediation action:",
  "options": [
    {"id": "scale_hpa", "text": "Trigger immediate HPA horizontal scale-out"},
    {"id": "restart_pod", "text": "Restart failed pod container instance"},
    {"id": "increase_memory", "text": "Patch deployment memory limits from 2Gi to 4Gi"}
  ],
  "label": "increase_memory",
  "decision_type": "choice"
}
```

### Format 2: Kev / TypeSafe v7 Schema
```json
{
  "state": "Customer requesting immediate invoice refund for annual renewal.",
  "questions": {
    "routing": {
      "type": "choice",
      "instructions": "Route to the appropriate customer care tier:",
      "criteria": {
        "tier1": "Standard billing support",
        "retention": "Customer success / retention queue",
        "finance": "Manual executive finance approval"
      },
      "label": "retention"
    }
  }
}
```

---

## 📊 Step 1: Plan Dataset Sizing

Before fine-tuning, run the dataset sizing advisor to assess data adequacy, cardinality distribution, expected accuracy, and compute budget:

```bash
# Via CLI
tamev-finetune --plan-size --data path/to/dataset.jsonl

# Or via Python SDK
from tamev import plan_dataset_size

report = plan_dataset_size("path/to/dataset.jsonl")
print(report["adequacy"], report["expected_accuracy"])
```

### Sizing Benchmarks:
| Sample Count | Adequacy Level | Expected Accuracy | Recommended Tier |
|---|---|---|---|
| `< 50` | Early Prototype | 60% - 70% | `nano` |
| `50 - 250` | Domain Specialization | 70% - 82% | `nano` or `micro` |
| `250 - 1,000` | Solid Production | 80% - 90% | `micro` or `small` |
| `> 1,000` | Enterprise Grade | 90% - 97%+ | `small` or `medium` |

---

## 🚀 Step 2: 1-Command Fine-Tuning

Run fine-tuning with automatic train/val split, quality filtering, early stopping, and multi-format export:

### Shell Command
```bash
# Instant fine-tune on CPU or Apple Silicon
tamev-finetune \
  --data path/to/dataset.jsonl \
  --tier nano \
  --epochs 3 \
  --batch-size 8 \
  --device auto \
  --output-dir ./runs/my_model
```

### Python SDK
```python
from tamev import finetune

results = finetune(
    data="path/to/dataset.jsonl",
    tier="micro",  # nano (14.4M), micro (22.8M), small (149M), medium (873M backbone + 0.5M head)
    epochs=4,
    batch_size=16,
    device="auto",  # "mps" for Apple Silicon, "cuda", or "cpu"
    output_dir="./runs/custom_decision_model",
    export_formats=["pytorch_fp32", "onnx_int8", "coreml", "mlx"],
)

print(
    f"Trained in {results['elapsed_sec']:.2f}s | Calibrated Temp: {results['calibrated_temperature']:.3f}"
)
```

---

## 📦 Step 3: Deployment & Serving

The fine-tuning pipeline automatically produces optimized production artifacts in `<output_dir>/exported/`:

1. **`deployment_manifest.json`**: Model metadata, parameter counts, SHA-256 signatures, and calibrated temperature.
2. **`tamev_<tier>_fp32.pt`**: Native PyTorch weights with calibrated temperature.
3. **`tamev_<tier>_int8.onnx`**: Quantized INT8 ONNX graph (< 15MB) for sub-5ms CPU execution.
4. **`tamev_<tier>.mlpackage`**: Apple CoreML model for Apple Neural Engine (ANE) on iOS/macOS.
5. **`tamev_<tier>_mlx/`**: Native Apple Silicon MLX weights (`weights.npz` + config) for unified memory serving at ~0.4ms.

### In-Process High-Speed Serving Example:
```python
from decision_engine.inference.direct_client import TypeSafeDirectClient

client = TypeSafeDirectClient.from_checkpoint("./runs/my_model/exported/tamev_nano_fp32.pt")
response = client.request(
    {
        "state": "High latency on payment gateway",
        "questions": {
            "action": {
                "type": "choice",
                "criteria": {
                    "reroute": "Switch to secondary payment gateway",
                    "drop": "Rate limit incoming transactions",
                },
            }
        },
    }
)
print("Decision:", response["choices"]["action"]["decision"])
print("Calibrated Likelihood:", response["choices"]["action"]["probabilities"])
```
