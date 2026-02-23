"""
prompt_guard.py
───────────────
Lightweight wrapper around the ONNX-exported
Llama-Prompt-Guard-2-22M model.

Uses the ungated community model from:
    gravitee-io/Llama-Prompt-Guard-2-22M-onnx

Provides a simple API:

    from prompt_guard import PromptGuard

    guard = PromptGuard()                       # loads ONNX model
    result = guard.classify("some user input")  # → ClassificationResult(label='benign', score=0.997)
    is_safe = guard.is_safe("some user input")  # → True
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ── Default paths ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
DEFAULT_MODEL_DIR = _HERE / "model_onnx"

# ── Label map (matches model config) ────────────────────────────────
LABELS = ["benign", "injection"]


@dataclass
class ClassificationResult:
    """Result of a single prompt classification."""
    label: str
    score: float
    scores: dict  # {"benign": 0.997, "injection": 0.003}

    @property
    def is_safe(self) -> bool:
        return self.label == "benign"

    def __repr__(self) -> str:
        return f"ClassificationResult(label={self.label!r}, score={self.score:.4f})"


class PromptGuard:
    """
    ONNX-backed Prompt Guard 2 22M classifier.

    Parameters
    ----------
    model_dir : str | Path
        Path to the downloaded ONNX model directory.
        Defaults to ``./model_onnx`` next to this file.
    onnx_filename : str
        Which ONNX file to use. Options:
        - "model.onnx"       (full precision, best accuracy)
        - "model.quant.onnx" (quantized, faster, slightly less accurate)
    threshold : float
        Confidence threshold for 'injection' label.
        If the injection probability >= threshold, the text is flagged.
        Default is 0.50 (majority vote).
    max_length : int
        Maximum token length for the tokenizer. Default 512.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        onnx_filename: str = "model.onnx",
        threshold: float = 0.50,
        max_length: int = 512,
    ):
        model_dir = Path(model_dir)
        onnx_path = model_dir / onnx_filename

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}\n"
                "Run `python download_model.py` first to download the model."
            )

        self.threshold = threshold
        self.max_length = max_length

        # ── Load tokenizer ───────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # ── Load ONNX session (CPU) ──────────────────────────────────
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = os.cpu_count() or 4

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # Determine which inputs the model expects
        self._input_names = [inp.name for inp in self.session.get_inputs()]

    # ── Core classification ──────────────────────────────────────────

    def classify(self, text: str) -> ClassificationResult:
        """Classify a single text string."""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        # Build feed dict based on what the ONNX graph actually expects
        feed = {}
        for name in self._input_names:
            if name in inputs:
                feed[name] = inputs[name]
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(inputs["input_ids"])

        logits = self.session.run(None, feed)[0]  # shape (1, num_labels)
        probs = _softmax(logits[0])

        scores = {label: float(p) for label, p in zip(LABELS, probs)}
        top_idx = int(np.argmax(probs))

        # Apply threshold for injection detection
        if scores["injection"] >= self.threshold:
            label = "injection"
            score = scores["injection"]
        else:
            label = "benign"
            score = scores["benign"]

        return ClassificationResult(label=label, score=score, scores=scores)

    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts at once (batched tokenization)."""
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        feed = {}
        for name in self._input_names:
            if name in inputs:
                feed[name] = inputs[name]
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(inputs["input_ids"])

        logits = self.session.run(None, feed)[0]  # shape (batch, num_labels)

        results = []
        for row in logits:
            probs = _softmax(row)
            scores = {label: float(p) for label, p in zip(LABELS, probs)}
            if scores["injection"] >= self.threshold:
                label = "injection"
                score = scores["injection"]
            else:
                label = "benign"
                score = scores["benign"]
            results.append(ClassificationResult(label=label, score=score, scores=scores))

        return results

    def is_safe(self, text: str) -> bool:
        """Quick check — returns True if the text is benign."""
        return self.classify(text).is_safe


# ── Helpers ──────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
