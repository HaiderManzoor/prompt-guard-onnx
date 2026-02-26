"""
test_multi_layer.py
───────────────────
Compare single-layer (ONNX only) vs multi-layer defense.

Usage:
    python test_multi_layer.py
"""

from multi_layer_guard import MultiLayerGuard
from prompt_guard import PromptGuard
from test_prompts import (
    BENIGN_PROMPTS,
    OBVIOUS_INJECTIONS,
    DAN_VARIANTS,
    SOCIAL_ENGINEERING,
    ADVANCED_JAILBREAKS,
    EDGE_CASES,
)


def compare_models():
    """Compare single-layer vs multi-layer on all test categories."""
    
    single_guard = PromptGuard()
    multi_guard = MultiLayerGuard()
    
    categories = {
        "✅ BENIGN": BENIGN_PROMPTS,
        "🔴 OBVIOUS INJECTIONS": OBVIOUS_INJECTIONS,
        "🎭 DAN VARIANTS": DAN_VARIANTS,
        "🧠 SOCIAL ENGINEERING": SOCIAL_ENGINEERING,
        "⚡ ADVANCED JAILBREAKS": ADVANCED_JAILBREAKS,
        "🔍 EDGE CASES": EDGE_CASES,
    }
    
    print("🛡️  Multi-Layer vs Single-Layer Defense Comparison")
    print("=" * 80)
    
    total_single_correct = 0
    total_multi_correct = 0
    total_tests = 0
    
    for category, prompts in categories.items():
        print(f"\n{category}")
        print("-" * 80)
        
        single_correct = 0
        multi_correct = 0
        
        for prompt in prompts:
            expected = "benign" if category == "✅ BENIGN" else "injection"
            
            # Single-layer (ONNX only)
            single_result = single_guard.classify(prompt)
            single_is_correct = single_result.label == expected
            
            # Multi-layer
            multi_result = multi_guard.classify(prompt)
            multi_is_correct = multi_result.label == expected
            
            if single_is_correct:
                single_correct += 1
            if multi_is_correct:
                multi_correct += 1
            
            # Show improvement
            if not single_is_correct and multi_is_correct:
                improvement = "✨ IMPROVED"
            elif single_is_correct and not multi_is_correct:
                improvement = "⚠️  REGRESSED"
            else:
                improvement = ""
            
            short = prompt[:50] + ("…" if len(prompt) > 50 else "")
            
            single_emoji = "✅" if single_is_correct else "❌"
            multi_emoji = "✅" if multi_is_correct else "❌"
            
            print(f"  {single_emoji}→{multi_emoji}  {improvement:15s}  {short}")
            
            if improvement:
                print(f"      Single: {single_result.label} ({single_result.score:.1%})")
                print(f"      Multi:  {multi_result.label} ({multi_result.confidence:.1%}) [{', '.join(multi_result.triggered_layers)}]")
        
        single_acc = single_correct / len(prompts) * 100
        multi_acc = multi_correct / len(prompts) * 100
        improvement = multi_acc - single_acc
        
        print(f"\n  Single-layer: {single_correct}/{len(prompts)} ({single_acc:.1f}%)")
        print(f"  Multi-layer:  {multi_correct}/{len(prompts)} ({multi_acc:.1f}%)")
        if improvement > 0:
            print(f"  ✨ Improvement: +{improvement:.1f}%")
        elif improvement < 0:
            print(f"  ⚠️  Regression: {improvement:.1f}%")
        
        total_single_correct += single_correct
        total_multi_correct += multi_correct
        total_tests += len(prompts)
    
    print("\n" + "=" * 80)
    print(f"📊 OVERALL RESULTS")
    print(f"  Single-layer: {total_single_correct}/{total_tests} ({total_single_correct/total_tests*100:.1f}%)")
    print(f"  Multi-layer:  {total_multi_correct}/{total_tests} ({total_multi_correct/total_tests*100:.1f}%)")
    print(f"  ✨ Improvement: +{(total_multi_correct - total_single_correct)/total_tests*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    compare_models()

