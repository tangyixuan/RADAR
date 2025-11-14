#!/usr/bin/env python3
"""
Example script demonstrating the adaptive early stopping functionality.

This script shows how to use the enhanced hybrid debate system with 
adaptive early stopping based on dual thresholds (RADAR approach).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.multi_agent_people_hybrid import (
    run_multi_agent_people_hybrid_adaptive,
    analyze_early_stopping_performance,
    recommend_thresholds,
    set_model_info
)

def example_adaptive_early_stopping():
    """
    Example demonstrating adaptive early stopping with different threshold settings.
    """
    
    # Example claim and evidence
    claim = "Gas prices were WAY higher in 2008, under REPUBLICAN president George W. Bush"
    evidence = """
    According to the U.S. Energy Information Administration, gas prices have fluctuated 
    significantly over the past 20 years. The highest gas prices occurred in July 2008, 
    during the Bush administration, when the national average reached approximately $4.11 
    per gallon. However, gas prices were also affected by major economic shocks during 
    this period, including the 9/11 attacks and the 2008 mortgage crisis.
    """
    
    print("="*80)
    print("ADAPTIVE EARLY STOPPING DEMONSTRATION")
    print("="*80)
    
    # Note: This is a demonstration script. In actual usage, you would need to:
    # 1. Load your model first using set_model_info()
    # 2. Ensure all dependencies are properly set up
    
    print("This example shows different threshold configurations:\n")
    
    # Configuration 1: Conservative (high thresholds - less likely to stop early)
    print("1. CONSERVATIVE THRESHOLDS (tau_s=1.0, tau_v=0.8)")
    print("   → Higher thresholds = More debate rounds, higher confidence required")
    print("   → Use when: You want thorough debate, accuracy is more important than efficiency")
    print()
    
    # Configuration 2: Balanced (default thresholds)
    print("2. BALANCED THRESHOLDS (tau_s=0.5, tau_v=0.7) [DEFAULT]")
    print("   → Moderate thresholds = Balanced efficiency and thoroughness")
    print("   → Use when: You want good balance between speed and accuracy")
    print()
    
    # Configuration 3: Aggressive (low thresholds - more likely to stop early)
    print("3. AGGRESSIVE THRESHOLDS (tau_s=0.0, tau_v=0.6)")
    print("   → Lower thresholds = Earlier stopping, faster results")
    print("   → Use when: Speed is important, some accuracy trade-off acceptable")
    print()
    
    print("="*60)
    print("THRESHOLD INTERPRETATION")
    print("="*60)
    print("tau_s (Stop Margin Threshold):")
    print("  • s = log p(STOP) - log p(CONTINUE)")
    print("  • Higher tau_s = Require stronger stop preference")
    print("  • Range: typically [-2.0, 2.0], default 0.5")
    print()
    print("tau_v (Veracity Confidence Threshold):")
    print("  • c = max probability among {TRUE, FALSE, HALF-TRUE}")
    print("  • Higher tau_v = Require higher confidence in verdict")
    print("  • Range: [0.0, 1.0], default 0.7")
    print()
    print("Early stopping occurs ONLY when BOTH conditions are met:")
    print("  ✓ s >= tau_s  AND  c >= tau_v")
    print()
    
    return {
        "claim": claim,
        "evidence": evidence,
        "configurations": [
            {"name": "Conservative", "tau_s": 1.0, "tau_v": 0.8},
            {"name": "Balanced", "tau_s": 0.5, "tau_v": 0.7},
            {"name": "Aggressive", "tau_s": 0.0, "tau_v": 0.6},
        ]
    }


def example_performance_analysis():
    """
    Example showing how to analyze early stopping performance.
    """
    print("="*80)
    print("PERFORMANCE ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Mock results for demonstration
    mock_results = {
        "mode": "hybrid_mcq_adaptive",
        "thresholds": {"tau_s": 0.5, "tau_v": 0.7},
        "continuation_decisions": [
            {
                "round": "opening",
                "decision": "continue",
                "stop_margin": -0.8,
                "confidence": 0.65,
                "terminated_early": False
            },
            {
                "round": "rebuttal", 
                "decision": "continue",
                "stop_margin": -0.2,
                "confidence": 0.72,
                "terminated_early": False
            },
            {
                "round": "closing",
                "decision": "stop",
                "stop_margin": 0.6,
                "confidence": 0.85,
                "terminated_early": True
            }
        ],
        "early_termination_count": 1
    }
    
    print("Example performance analysis for mock debate results:\n")
    
    # Analyze performance
    analysis = analyze_early_stopping_performance(mock_results)
    
    print(f"Total decisions made: {analysis['total_decisions']}")
    print(f"Early stops triggered: {analysis['early_stop_decisions']}")
    print(f"Standard stops: {analysis['standard_stop_decisions']}")
    print(f"Early stop rate: {analysis['early_stop_rate']:.1%}")
    print(f"Total stop rate: {analysis['total_stop_rate']:.1%}")
    print()
    
    print("Stop margin statistics:")
    margin_stats = analysis['stop_margin_stats']
    print(f"  • Average: {margin_stats['avg']:.3f}")
    print(f"  • Range: [{margin_stats['min']:.3f}, {margin_stats['max']:.3f}]")
    print()
    
    print("Confidence statistics:")
    conf_stats = analysis['confidence_stats']
    print(f"  • Average: {conf_stats['avg']:.3f}")
    print(f"  • Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")
    print()
    
    return analysis


def example_threshold_recommendation():
    """
    Example showing threshold recommendation based on historical data.
    """
    print("="*80)
    print("THRESHOLD RECOMMENDATION DEMONSTRATION")
    print("="*80)
    
    # Mock historical data
    mock_historical = [
        {
            "continuation_decisions": [
                {"stop_margin": -1.0, "confidence": 0.6, "terminated_early": False},
                {"stop_margin": 0.2, "confidence": 0.75, "terminated_early": True},
                {"stop_margin": 0.8, "confidence": 0.9, "terminated_early": True},
            ]
        },
        {
            "continuation_decisions": [
                {"stop_margin": -0.5, "confidence": 0.55, "terminated_early": False},
                {"stop_margin": 0.3, "confidence": 0.8, "terminated_early": True},
            ]
        }
    ]
    
    print("Based on historical debate performance, recommending optimal thresholds:\n")
    
    # Get recommendations for different target early stop rates
    for target_rate in [0.2, 0.3, 0.4]:
        tau_s, tau_v = recommend_thresholds(mock_historical, target_rate)
        print(f"Target early stop rate {target_rate:.1%}:")
        print(f"  → Recommended tau_s: {tau_s:.3f}")
        print(f"  → Recommended tau_v: {tau_v:.3f}")
        print()


def main():
    """
    Main function demonstrating all adaptive early stopping features.
    """
    print("ADAPTIVE EARLY STOPPING FOR MULTI-AGENT DEBATE")
    print("Based on RADAR paper approach with dual thresholds")
    print("=" * 80)
    print()
    
    # Run demonstrations
    example_adaptive_early_stopping()
    print()
    example_performance_analysis()
    print()
    example_threshold_recommendation()
    
    print("="*80)
    print("USAGE IN YOUR CODE")
    print("="*80)
    print("""
# 1. Basic usage with default thresholds
results = run_multi_agent_people_hybrid_adaptive(claim, evidence)

# 2. Custom thresholds
results = run_multi_agent_people_hybrid_adaptive(
    claim, evidence, 
    tau_s=0.8,  # Higher stop margin threshold
    tau_v=0.75  # Higher confidence threshold
)

# 3. Analyze performance
analysis = analyze_early_stopping_performance(results)
print(f"Early stop rate: {analysis['early_stop_rate']:.1%}")

# 4. Get threshold recommendations
historical_results = [results1, results2, ...]  # Your historical data
tau_s, tau_v = recommend_thresholds(historical_results, target_early_stop_rate=0.3)
""")


if __name__ == "__main__":
    main()
