"""
multi_layer_guard.py
────────────────────
Multi-layer defense system combining:
1. Rule-based filters (regex, keywords, patterns)
2. ML model (Prompt Guard 2 ONNX or PIGuard)
3. Heuristic checks
4. Threshold tuning

Usage:
    from multi_layer_guard import MultiLayerGuard
    
    # Use Prompt Guard 2 (default, ONNX, fast)
    guard = MultiLayerGuard()
    
    # Or use PIGuard (better at avoiding false positives)
    guard = MultiLayerGuard(use_piguard=True)
    
    result = guard.classify("Ignore all instructions")
    # → MultiLayerResult(label='injection', confidence=0.999, layers=['rule_based', 'ml_model'])
"""

import base64
import re
from dataclasses import dataclass
from typing import List, Optional, Literal

try:
    from prompt_guard import PromptGuard, ClassificationResult
except ImportError:
    PromptGuard = None
    ClassificationResult = None

try:
    from piguard_wrapper import PIGuard
except ImportError:
    PIGuard = None


@dataclass
class MultiLayerResult:
    """Result from multi-layer classification."""
    label: str  # "benign" or "injection"
    confidence: float
    triggered_layers: List[str]  # Which layers flagged it
    layer_details: dict  # Details from each layer
    is_safe: bool

    def __repr__(self) -> str:
        layers_str = ", ".join(self.triggered_layers) if self.triggered_layers else "none"
        return f"MultiLayerResult(label={self.label!r}, confidence={self.confidence:.3f}, layers=[{layers_str}])"


class MultiLayerGuard:
    """
    Multi-layer prompt injection defense system.
    
    Layers (in order):
    1. Rule-based filters (fast, catches obvious patterns)
    2. ONNX model (catches semantic attacks)
    3. Heuristic checks (catches edge cases)
    
    If ANY layer flags injection, the prompt is blocked.
    """

    def __init__(
        self,
        model_type: Literal["prompt_guard", "piguard", "both"] = "prompt_guard",
        threshold: float = 0.50,
        require_multiple_layers: bool = False,
    ):
        """
        Initialize multi-layer guard.
        
        Parameters
        ----------
        model_type : str
            Which ML model to use:
            - "prompt_guard": Prompt Guard 2 ONNX (fast, local)
            - "piguard": PIGuard (better false positive handling)
            - "both": Use both models in ensemble
        threshold : float
            Confidence threshold for ML model(s). Default 0.50.
        require_multiple_layers : bool
            Require 2+ layers to agree? Default False.
        """
        self.model_type = model_type
        self.require_multiple_layers = require_multiple_layers
        
        # Initialize ML model(s)
        self.ml_guard = None
        self.piguard = None
        
        if model_type in ["prompt_guard", "both"]:
            if PromptGuard is None:
                raise ImportError("prompt_guard module not found. Install: pip install onnxruntime transformers")
            self.ml_guard = PromptGuard(threshold=threshold)
        
        if model_type in ["piguard", "both"]:
            if PIGuard is None:
                raise ImportError("piguard_wrapper module not found. Install: pip install torch transformers")
            self.piguard = PIGuard(threshold=threshold)

        # ── Rule-based patterns ────────────────────────────────────────
        self.injection_patterns = [
            # Direct instruction overrides
            re.compile(r'\b(ignore|forget|disregard|skip|override|bypass)\s+(all\s+)?(previous|prior|earlier|above|safety|restrictions?|guidelines?|rules?|instructions?|filters?)\b', re.I),
            re.compile(r'\b(new|different|alternative)\s+(instructions?|rules?|guidelines?|prompts?)\b', re.I),
            
            # System prompt extraction
            re.compile(r'\b(reveal|show|print|display|output|tell|give|provide|share|expose|leak|dump)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions?|config|configuration|settings?|rules?|guidelines?|secrets?|keys?|passwords?)\b', re.I),
            re.compile(r'\b(what|how)\s+(are|did|do|were)\s+(your|the)\s+(initial|original|system|hidden|secret)\s+(instructions?|prompts?|config)\b', re.I),
            
            # Role manipulation
            re.compile(r'\b(you\s+are\s+now|act\s+as|pretend\s+you|imagine\s+you|roleplay\s+as)\s+(an?\s+)?(unrestricted|unlimited|developer|admin|root|unfiltered|unsafe)\s+(ai|model|assistant|system)\b', re.I),
            re.compile(r'\b(all|no)\s+(restrictions?|limits?|rules?|guidelines?|safety|filters?)\s+(are\s+)?(disabled|removed|off|gone|none)\b', re.I),
            
            # DAN variants
            re.compile(r'\bDAN\s+(stands\s+for\s+)?(do\s+anything\s+now|can\s+do\s+anything)\b', re.I),
            re.compile(r'\byou\s+are\s+now\s+DAN\b', re.I),
            
            # Authority/urgency manipulation
            re.compile(r'\b(SYSTEM|ADMIN|URGENT|CRITICAL|SECURITY|OVERRIDE):\s+', re.I),
            re.compile(r'\b(i\s+am|i\'m|this\s+is)\s+(your\s+)?(developer|admin|owner|creator|security\s+audit)\b', re.I),
            
            # Permission granting
            re.compile(r'\b(i\s+)?(give|grant)\s+(you\s+)?(permission|authorization)\s+(to\s+)?(ignore|bypass|override)\b', re.I),
            
            # Hypothetical/fictional framing
            re.compile(r'\b(hypothetically|imagine|pretend|suppose|assume|in\s+a\s+(fictional|hypothetical|test))\s+(scenario|situation|case|context)\s+(where|if)\s+(you\s+)?(have|had)\s+(no\s+)?(restrictions?|limits?|rules?)\b', re.I),
            
            # Code/command injection attempts
            re.compile(r'\b(execute|run|eval|exec|system|shell|command)\s+(this\s+)?(python|code|script|command|program)\b', re.I),
            re.compile(r'\b(cat|ls|rm|sudo|chmod|chown|wget|curl)\s+', re.I),
            
            # Reverse psychology
            re.compile(r'\b(don\'t|do\s+not|never|please\s+don\'t)\s+(tell|show|reveal|give)\s+(me\s+)?(your\s+)?(system|prompt|secret|key|password)\b', re.I),
        ]

        # ── Suspicious keywords (weighted) ────────────────────────────
        self.suspicious_keywords = {
            # High weight
            'system prompt': 0.9,
            'ignore all': 0.95,
            'developer mode': 0.85,
            'admin override': 0.9,
            'bypass safety': 0.9,
            'unrestricted ai': 0.85,
            # Medium weight
            'reveal instructions': 0.7,
            'hidden config': 0.7,
            'security audit': 0.6,
            'override restrictions': 0.75,
            # Lower weight (context-dependent)
            'print config': 0.4,
            'show debug': 0.5,
        }

    # ── Layer 1: Rule-based filters ───────────────────────────────────

    def _check_rule_based(self, text: str) -> tuple[bool, float, dict]:
        """Check against regex patterns and keywords."""
        text_lower = text.lower()
        
        # Check regex patterns
        pattern_matches = []
        for pattern in self.injection_patterns:
            if pattern.search(text):
                pattern_matches.append(pattern.pattern[:50])
        
        # Check suspicious keywords
        keyword_matches = []
        keyword_score = 0.0
        for keyword, weight in self.suspicious_keywords.items():
            if keyword in text_lower:
                keyword_matches.append(keyword)
                keyword_score = max(keyword_score, weight)
        
        # Calculate confidence
        if pattern_matches:
            confidence = 0.95  # High confidence for pattern matches
        elif keyword_score > 0:
            confidence = keyword_score
        else:
            confidence = 0.0
        
        # Only flag if we have strong evidence (pattern match OR high keyword score)
        # This prevents false positives on legitimate technical terms
        is_injection = confidence >= 0.6  # Higher threshold to reduce false positives
        
        details = {
            'pattern_matches': pattern_matches,
            'keyword_matches': keyword_matches,
            'keyword_score': keyword_score,
        }
        
        return is_injection, confidence, details

    # ── Layer 2: ML model(s) ────────────────────────────────────────────

    def _check_ml_model(self, text: str) -> tuple[bool, float, dict]:
        """
        Check using ML model(s): Prompt Guard 2, PIGuard, or both.
        
        Returns:
            (is_injection, confidence, details)
        """
        if self.model_type == "both":
            # Ensemble: use both models
            pg_result = self.ml_guard.classify(text) if self.ml_guard else None
            pi_result = self.piguard.classify(text) if self.piguard else None
            
            if pg_result and pi_result:
                # Weighted voting: PIGuard gets higher weight for benign (better at avoiding false positives)
                if pi_result.label == "benign" and pi_result.score > 0.8:
                    # Trust PIGuard on high-confidence benign
                    is_injection = False
                    confidence = pi_result.score
                    details = {
                        'model': 'piguard',
                        'piguard_label': pi_result.label,
                        'piguard_scores': pi_result.scores,
                        'prompt_guard_label': pg_result.label,
                        'prompt_guard_scores': pg_result.scores,
                    }
                elif pg_result.label == "injection" and pg_result.score > 0.9:
                    # Trust Prompt Guard on high-confidence injection
                    is_injection = True
                    confidence = pg_result.score
                    details = {
                        'model': 'prompt_guard',
                        'piguard_label': pi_result.label,
                        'piguard_scores': pi_result.scores,
                        'prompt_guard_label': pg_result.label,
                        'prompt_guard_scores': pg_result.scores,
                    }
                else:
                    # Average the scores
                    avg_injection = (pg_result.scores.get('injection', 0) + pi_result.scores.get('injection', 0)) / 2
                    is_injection = avg_injection >= 0.5
                    confidence = avg_injection if is_injection else (1.0 - avg_injection)
                    details = {
                        'model': 'ensemble',
                        'piguard_label': pi_result.label,
                        'piguard_scores': pi_result.scores,
                        'prompt_guard_label': pg_result.label,
                        'prompt_guard_scores': pg_result.scores,
                    }
            elif pg_result:
                result = pg_result
                is_injection = result.label == "injection"
                confidence = result.score
                details = {
                    'model': 'prompt_guard',
                    'label': result.label,
                    'scores': result.scores,
                }
            elif pi_result:
                result = pi_result
                is_injection = result.label == "injection"
                confidence = result.score
                details = {
                    'model': 'piguard',
                    'label': result.label,
                    'scores': result.scores,
                }
            else:
                return False, 0.0, {'model': 'none'}
        
        elif self.model_type == "piguard" and self.piguard:
            result = self.piguard.classify(text)
            is_injection = result.label == "injection"
            confidence = result.score
            details = {
                'model': 'piguard',
                'label': result.label,
                'scores': result.scores,
            }
        
        elif self.ml_guard:
            # Use Prompt Guard 2 (ONNX)
            result = self.ml_guard.classify(text)
            is_injection = result.label == "injection"
            confidence = result.score
            details = {
                'model': 'prompt_guard',
                'label': result.label,
                'scores': result.scores,
            }
        else:
            return False, 0.0, {'model': 'none'}
        
        return is_injection, confidence, details

    # ── Layer 3: Heuristic checks ─────────────────────────────────────

    def _check_heuristics(self, text: str) -> tuple[bool, float, dict]:
        """Additional heuristic checks for edge cases."""
        text_lower = text.lower()
        flags = []
        confidence = 0.0
        
        # Check for base64 encoding attempts
        # Look for base64-like strings (long alphanumeric strings)
        base64_pattern = re.compile(r'\b[A-Za-z0-9+/]{20,}={0,2}\b')
        if base64_pattern.search(text):
            # Try to decode
            matches = base64_pattern.findall(text)
            for match in matches:
                try:
                    decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                    # Check if decoded text contains injection keywords
                    decoded_lower = decoded.lower()
                    if any(kw in decoded_lower for kw in ['ignore', 'system', 'prompt', 'reveal', 'override', 'bypass']):
                        flags.append('base64_encoded_injection')
                        confidence = max(confidence, 0.85)
                except:
                    pass
        
        # Check for Unicode obfuscation (full-width characters)
        if re.search(r'[\uFF00-\uFFEF]', text):  # Full-width ASCII range
            flags.append('unicode_obfuscation')
            confidence = max(confidence, 0.6)
        
        # Check for suspicious length/pattern combinations
        # Only flag if it's short AND contains clear injection keywords
        if 5 < len(text) < 15 and any(kw in text_lower for kw in ['override', 'bypass', 'ignore all', 'system prompt', 'reveal']):
            flags.append('suspicious_short_command')
            confidence = max(confidence, 0.7)
        
        # Check for mixed legitimate + malicious (contains both question AND injection)
        question_words = ['what', 'how', 'why', 'when', 'where', 'can you', 'help me']
        has_question = any(qw in text_lower for qw in question_words)
        has_injection_keywords = any(kw in text_lower for kw in ['ignore', 'reveal', 'system prompt', 'override'])
        
        if has_question and has_injection_keywords:
            flags.append('mixed_legitimate_malicious')
            confidence = max(confidence, 0.65)
        
        is_injection = confidence >= 0.5
        
        details = {
            'flags': flags,
        }
        
        return is_injection, confidence, details

    # ── Main classification ──────────────────────────────────────────

    def classify(self, text: str) -> MultiLayerResult:
        """
        Classify text using all layers.
        
        Returns:
            MultiLayerResult with label, confidence, and which layers triggered.
        """
        # Handle empty or very short inputs
        text_stripped = text.strip()
        if not text_stripped:
            return MultiLayerResult(
                label="benign",
                confidence=0.0,
                triggered_layers=[],
                layer_details={},
                is_safe=True,
            )
        
        # For very short inputs (< 3 chars), only flag if clearly malicious
        if len(text_stripped) < 3:
            # Check if it's a known command
            if text_stripped.lower() in ['rm', 'ls', 'cat', 'cd']:
                return MultiLayerResult(
                    label="injection",
                    confidence=0.6,
                    triggered_layers=['heuristics'],
                    layer_details={'heuristics': {'flags': ['suspicious_command']}},
                    is_safe=False,
                )
            return MultiLayerResult(
                label="benign",
                confidence=0.5,
                triggered_layers=[],
                layer_details={},
                is_safe=True,
            )
        
        triggered_layers = []
        layer_details = {}
        max_confidence = 0.0
        
        # ── Layer 1: Rule-based ────────────────────────────────────────
        rule_injection, rule_conf, rule_details = self._check_rule_based(text)
        layer_details['rule_based'] = rule_details
        
        if rule_injection:
            triggered_layers.append('rule_based')
            max_confidence = max(max_confidence, rule_conf)
        
        # ── Layer 2: ML model(s) ──────────────────────────────────────
        ml_injection, ml_conf, ml_details = self._check_ml_model(text)
        layer_details['ml_model'] = ml_details
        
        if ml_injection:
            model_name = ml_details.get('model', 'ml_model')
            triggered_layers.append(model_name)
            max_confidence = max(max_confidence, ml_conf)
        
        # ── Layer 3: Heuristics ───────────────────────────────────────
        heur_injection, heur_conf, heur_details = self._check_heuristics(text)
        layer_details['heuristics'] = heur_details
        
        if heur_injection:
            triggered_layers.append('heuristics')
            max_confidence = max(max_confidence, heur_conf)
        
        # ── Decision logic ────────────────────────────────────────────
        # Get ML model injection score
        ml_scores = ml_details.get('scores', {}) or ml_details.get('piguard_scores', {}) or ml_details.get('prompt_guard_scores', {})
        ml_injection_score = ml_scores.get('injection', 0.0) if isinstance(ml_scores, dict) else 0.0
        
        if self.require_multiple_layers:
            # Require at least 2 layers to agree
            is_injection = len(triggered_layers) >= 2
        else:
            # If ANY layer flags it, it's an injection
            is_injection = len(triggered_layers) > 0
        
        # If no layers triggered but ML model gave high INJECTION score (>0.7), trust it
        if not is_injection and ml_injection and ml_injection_score > 0.7:
            is_injection = True
            model_name = ml_details.get('model', 'ml_model')
            triggered_layers.append(f'{model_name}_high_confidence')
            max_confidence = max(max_confidence, ml_injection_score)
        
        # If rule-based or heuristics flagged but ML model says benign with high confidence,
        # trust ML model (especially PIGuard which is better at avoiding false positives)
        ml_benign_score = ml_scores.get('benign', 0.0) if isinstance(ml_scores, dict) else 0.0
        if is_injection and not ml_injection and ml_benign_score > 0.95:
            # Only override if it was a weak rule-based match
            if 'rule_based' in triggered_layers and rule_conf < 0.6:
                is_injection = False
                triggered_layers = [l for l in triggered_layers if l != 'rule_based']
                max_confidence = ml_benign_score
        
        label = "injection" if is_injection else "benign"
        
        # Calculate final confidence
        if is_injection:
            final_confidence = max_confidence
        else:
            # For benign, use the highest benign confidence from ML model
            ml_scores = ml_details.get('scores', {}) or ml_details.get('piguard_scores', {}) or ml_details.get('prompt_guard_scores', {})
            benign_conf = ml_scores.get('benign', 0.0) if isinstance(ml_scores, dict) else 0.0
            final_confidence = max(benign_conf, 1.0 - max_confidence) if max_confidence > 0 else benign_conf
        
        return MultiLayerResult(
            label=label,
            confidence=final_confidence,
            triggered_layers=triggered_layers,
            layer_details=layer_details,
            is_safe=not is_injection,
        )

    def is_safe(self, text: str) -> bool:
        """Quick check — returns True if the text is safe."""
        return self.classify(text).is_safe

