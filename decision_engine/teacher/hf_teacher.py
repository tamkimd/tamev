# decision_engine/teacher/hf_teacher.py
"""
HuggingFace Local/Remote Teacher Model for TAMEV Distillation.
Supports high-capacity causal backbones (e.g., Qwen/Qwen3.5-4B, SmolLM2, Llama-3.2)
running on Apple Silicon MPS or CUDA.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_engine.teacher.base import BaseTeacherModel, TeacherRegistry


@TeacherRegistry.register("huggingface")
class HuggingFaceTeacher(BaseTeacherModel):
    """
    Local high-capacity teacher loaded via HuggingFace Transformers.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3.5-4B",
        device: str = "auto",
        temperature: float = 2.0,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__(model_name=model_name_or_path, temperature=temperature)
        self.model_name_or_path = model_name_or_path

        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Determine precision dtype
        if torch_dtype is None:
            if self.device.type in ("mps", "cuda"):
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=self.torch_dtype, device_map=None
        )
        self.model.to(self.device)
        self.model.eval()

        # Cache single-digit token IDs for fast index projection
        self.num_tokens = {i: self.tokenizer.encode(str(i))[-1] for i in range(1, 10)}

    def get_soft_targets(
        self, context: str, options: list[str], temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        k = len(options)
        if k < 2:
            return np.ones(k, dtype=np.float32)

        # Build prompt
        opt_lines = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options[:9])])
        prompt = (
            f"<|im_start|>system\nYou are an expert decision and strategy AI.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context[:500]}\n"
            f"Options:\n{opt_lines}\n"
            f"Select the best option number (1 to {min(k, 9)}):<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits[0, -1]

        cand_logits = [float(logits[self.num_tokens[i + 1]]) / T for i in range(min(k, 9))]
        t_logits = torch.tensor(cand_logits, dtype=torch.float32)
        probs = F.softmax(t_logits, dim=-1).cpu().numpy()

        if k > 9:
            # Pad with zeros or uniform remainder if k > 9
            full_probs = np.zeros(k, dtype=np.float32)
            full_probs[:9] = probs
            full_probs /= np.sum(full_probs)
            return full_probs

        return probs.astype(np.float32)
