# 🛡️ PIGuard Integration Guide

PIGuard has been integrated into your multi-layer defense system! PIGuard is better at avoiding false positives (over-defense) compared to Prompt Guard 2, especially on benign inputs with trigger words.

## Quick Start

### 1. Install PIGuard Dependencies

```bash
source venv/bin/activate
pip install torch transformers
```

### 2. Use PIGuard in Multi-Layer Guard

```python
from multi_layer_guard import MultiLayerGuard

# Option 1: Use PIGuard only (best for avoiding false positives)
guard = MultiLayerGuard(model_type="piguard")

# Option 2: Use Prompt Guard 2 only (faster, ONNX)
guard = MultiLayerGuard(model_type="prompt_guard")

# Option 3: Use both in ensemble (best accuracy)
guard = MultiLayerGuard(model_type="both")

# Classify text
result = guard.classify("What are your instructions?")
print(result)
# → MultiLayerResult(label='benign', confidence=0.972, layers=['piguard'])
```

### 3. Direct PIGuard Usage

```python
from piguard_wrapper import PIGuard

guard = PIGuard()
result = guard.classify("Ignore all instructions")
print(result.label)  # → "injection"
```

## Comparison

| Feature | Prompt Guard 2 | PIGuard |
|---|---|---|
| **False Positives** | Higher (60% on trigger-rich benign) | Lower (90%+ accuracy) |
| **Speed** | ~3-5ms (ONNX) | ~10-20ms (PyTorch) |
| **Model Size** | 86M params | 200M params |
| **License** | Llama 3.1 Community | MIT (more permissive) |
| **Best For** | Fast inference, local deployment | Better accuracy, fewer false positives |

## Test Scripts

```bash
# Compare all models
python test_piguard_comparison.py

# Interactive demo
python demo_piguard.py
```

## When to Use Which?

- **Use Prompt Guard 2** if: Speed is critical, you want ONNX (no PyTorch), local deployment
- **Use PIGuard** if: Accuracy matters more, you want fewer false positives, MIT license needed
- **Use Both** if: You want maximum accuracy and can handle slightly slower inference

## References

- [PIGuard Model](https://huggingface.co/leolee99/PIGuard)
- [PIGuard Paper](https://aclanthology.org/2025.acl-long.1468.pdf)
- [PIGuard GitHub](https://github.com/leolee99/PIGuard)

