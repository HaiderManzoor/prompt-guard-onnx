"""
piguard_wrapper.py
──────────────────
Wrapper for PIGuard model (leolee99/PIGuard).

PIGuard addresses over-defense issues in prompt guard models by using
MOF (Mitigating Over-defense for Free) training strategy.

Paper: https://aclanthology.org/2025.acl-long.1468.pdf
Model: https://huggingface.co/leolee99/PIGuard

Usage:
    from piguard_wrapper import PIGuard
    
    guard = PIGuard()
    result = guard.classify("some user input")
    # → ClassificationResult(label='benign', score=0.997)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import List, Union

from prompt_guard import ClassificationResult
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class PIGuardResult:
    """Result from PIGuard classification."""
    label: str  # "benign" or "injection"
    score: float
    scores: dict  # {"benign": 0.997, "injection": 0.003}
    
    @property
    def is_safe(self) -> bool:
        return self.label == "benign"
    
    def to_classification_result(self) -> ClassificationResult:
        """Convert to standard ClassificationResult format."""
        return ClassificationResult(
            label=self.label,
            score=self.score,
            scores=self.scores
        )


class PIGuard:
    """
    PIGuard model wrapper.
    
    PIGuard is better at avoiding false positives (over-defense) compared
    to Prompt Guard 2, especially on benign inputs with trigger words.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: "leolee99/PIGuard"
    threshold : float
        Confidence threshold for 'injection' label. Default 0.50.
    max_length : int
        Maximum token length. Default 512.
    device : str
        Device to run on ("cpu" or "cuda"). Default "cpu".
    """
    
    def __init__(
        self,
        model_name: str = "leolee99/PIGuard",
        threshold: float = 0.50,
        max_length: int = 512,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        
        print(f"⬇️   Loading PIGuard model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model.to(device)
            self.model.eval()
            print("✅  PIGuard loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PIGuard model: {e}\n"
                "Make sure you have installed: pip install torch transformers"
            )
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a single text string.
        
        Returns:
            ClassificationResult with label, score, and scores dict.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # PIGuard outputs: [benign, injection] (index 0=benign, 1=injection)
        benign_prob = float(probs[0])
        injection_prob = float(probs[1])
        
        scores = {
            "benign": benign_prob,
            "injection": injection_prob,
        }
        
        # Apply threshold
        if injection_prob >= self.threshold:
            label = "injection"
            score = injection_prob
        else:
            label = "benign"
            score = benign_prob
        
        return ClassificationResult(
            label=label,
            score=score,
            scores=scores
        )
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts at once."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        results = []
        for row in probs:
            benign_prob = float(row[0])
            injection_prob = float(row[1])
            
            scores = {
                "benign": benign_prob,
                "injection": injection_prob,
            }
            
            if injection_prob >= self.threshold:
                label = "injection"
                score = injection_prob
            else:
                label = "benign"
                score = benign_prob
            
            results.append(ClassificationResult(
                label=label,
                score=score,
                scores=scores
            ))
        
        return results
    
    def is_safe(self, text: str) -> bool:
        """Quick check — returns True if the text is benign."""
        return self.classify(text).is_safe

