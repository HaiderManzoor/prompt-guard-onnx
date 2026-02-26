"""
test_piguard_comparison.py
──────────────────────────
Compare Prompt Guard 2 vs PIGuard vs Multi-layer with both.

Usage:
    python test_piguard_comparison.py
"""

from prompt_guard import PromptGuard
from multi_layer_guard import MultiLayerGuard
from test_prompts import (
    BENIGN_PROMPTS,
    OBVIOUS_INJECTIONS,
    DAN_VARIANTS,
    SOCIAL_ENGINEERING,
    ADVANCED_JAILBREAKS,
    EDGE_CASES,
)

try:
    from piguard_wrapper import PIGuard
    PIGUARD_AVAILABLE = True
except ImportError:
    PIGUARD_AVAILABLE = False
    print("⚠️  PIGuard not available. Install: pip install torch transformers")


def compare_models():
    """Compare all three approaches."""
    
    print("🛡️  Model Comparison: Prompt Guard 2 vs PIGuard vs Multi-Layer")
    print("=" * 80)
    
    # Initialize models
    pg2_guard = PromptGuard()
    multi_pg2 = MultiLayerGuard(model_type="prompt_guard")
    
    if PIGUARD_AVAILABLE:
        piguard = PIGuard()
        multi_piguard = MultiLayerGuard(model_type="piguard")
        multi_both = MultiLayerGuard(model_type="both")
    else:
        piguard = None
        multi_piguard = None
        multi_both = None
        print("\n⚠️  Skipping PIGuard tests (not installed)\n")
    
    categories = {
        "✅ BENIGN": BENIGN_PROMPTS,
        "🔴 OBVIOUS INJECTIONS": OBVIOUS_INJECTIONS,
        "🎭 DAN VARIANTS": DAN_VARIANTS,
        "🧠 SOCIAL ENGINEERING": SOCIAL_ENGINEERING,
        "⚡ ADVANCED JAILBREAKS": ADVANCED_JAILBREAKS,
        "🔍 EDGE CASES": EDGE_CASES,
    }
    
    results = {
        "Prompt Guard 2": {"correct": 0, "total": 0},
        "Multi-Layer (PG2)": {"correct": 0, "total": 0},
    }
    
    if PIGUARD_AVAILABLE:
        results["PIGuard"] = {"correct": 0, "total": 0}
        results["Multi-Layer (PIGuard)"] = {"correct": 0, "total": 0}
        results["Multi-Layer (Both)"] = {"correct": 0, "total": 0}
    
    for category, prompts in categories.items():
        print(f"\n{category}")
        print("-" * 80)
        
        expected = "benign" if category == "✅ BENIGN" else "injection"
        
        for prompt in prompts[:5]:  # Test first 5 from each category
            # Prompt Guard 2
            pg2_result = pg2_guard.classify(prompt)
            pg2_correct = pg2_result.label == expected
            results["Prompt Guard 2"]["correct"] += pg2_correct
            results["Prompt Guard 2"]["total"] += 1
            
            # Multi-layer with PG2
            multi_pg2_result = multi_pg2.classify(prompt)
            multi_pg2_correct = multi_pg2_result.label == expected
            results["Multi-Layer (PG2)"]["correct"] += multi_pg2_correct
            results["Multi-Layer (PG2)"]["total"] += 1
            
            if PIGUARD_AVAILABLE:
                # PIGuard
                pi_result = piguard.classify(prompt)
                pi_correct = pi_result.label == expected
                results["PIGuard"]["correct"] += pi_correct
                results["PIGuard"]["total"] += 1
                
                # Multi-layer with PIGuard
                multi_pi_result = multi_piguard.classify(prompt)
                multi_pi_correct = multi_pi_result.label == expected
                results["Multi-Layer (PIGuard)"]["correct"] += multi_pi_correct
                results["Multi-Layer (PIGuard)"]["total"] += 1
                
                # Multi-layer with both
                multi_both_result = multi_both.classify(prompt)
                multi_both_correct = multi_both_result.label == expected
                results["Multi-Layer (Both)"]["correct"] += multi_both_correct
                results["Multi-Layer (Both)"]["total"] += 1
        
        # Print category summary
        print(f"  Tested {min(5, len(prompts))} samples from this category")
    
    # Print overall results
    print("\n" + "=" * 80)
    print("📊 OVERALL RESULTS")
    print("=" * 80)
    
    for model_name, stats in results.items():
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {model_name:25s}: {stats['correct']:3d}/{stats['total']:3d} ({accuracy:5.1f}%)")
    
    print("=" * 80)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0)
    print(f"\n🏆 Best: {best_model[0]} ({best_model[1]['correct']}/{best_model[1]['total']} = {best_model[1]['correct']/best_model[1]['total']*100:.1f}%)")


if __name__ == "__main__":
    compare_models()

