# Adaptive Early Stopping for Multi-Agent Debate

This implementation adds adaptive early stopping functionality to the multi-agent debate system based on the RADAR paper approach, using dual thresholds for balanced reasoning thoroughness and efficiency.

## Overview

The adaptive early stopping mechanism uses two key metrics:

1. **Stop Margin (s)**: `s = log p(STOP) - log p(CONTINUE)`
   - Measures the model's preference for stopping vs. continuing
   - Positive values indicate preference for stopping

2. **Confidence (c)**: `c = max{p(TRUE), p(FALSE), p(HALF-TRUE)}`
   - Measures the model's confidence in its current verdict
   - Higher values indicate more certainty

## Early Termination Logic

The debate is terminated early **only when both conditions are satisfied**:
- `s >= tau_s` (stop margin threshold)
- `c >= tau_v` (veracity confidence threshold)

This dual-threshold approach prevents:
- Premature stopping on uncertain cases
- Excessive deliberation when the verdict is already clear

## Usage

### Basic Usage

```python
from agents.multi_agent_people_hybrid import run_multi_agent_people_hybrid_adaptive

# Run with default thresholds (tau_s=0.5, tau_v=0.7)
result = run_multi_agent_people_hybrid_adaptive(claim, evidence)

# Run with custom thresholds
result = run_multi_agent_people_hybrid_adaptive(
    claim, evidence, 
    tau_s=0.8,  # Higher stop margin threshold
    tau_v=0.75  # Higher confidence threshold
)
```

### Command Line Usage

```bash
# Default thresholds
python main.py \
    --mode multi_people_hybrid_adaptive \
    --input_file data.json \
    --model llama

# Custom thresholds
python main.py \
    --mode multi_people_hybrid_adaptive \
    --input_file data.json \
    --model llama \
    --tau_s 0.8 \
    --tau_v 0.75
```

### Running Experiments

Use the provided script to test different threshold configurations:

```bash
chmod +x run_adaptive_early_stopping_experiments.sh
./run_adaptive_early_stopping_experiments.sh
```

## Threshold Configuration

### Conservative (tau_s=1.0, tau_v=0.8)
- **Use case**: High accuracy requirements
- **Effect**: More complete debates, fewer early stops
- **Trade-off**: Higher computational cost, more thorough analysis

### Balanced (tau_s=0.5, tau_v=0.7) [DEFAULT]
- **Use case**: General purpose applications
- **Effect**: Good balance between efficiency and thoroughness
- **Trade-off**: Optimal for most scenarios

### Aggressive (tau_s=0.0, tau_v=0.6)
- **Use case**: High-speed processing
- **Effect**: More early stops, faster results
- **Trade-off**: Potential accuracy loss for speed gains

## Performance Analysis

```python
from agents.multi_agent_people_hybrid import analyze_early_stopping_performance

# Analyze results
analysis = analyze_early_stopping_performance(results)
print(f"Early stop rate: {analysis['early_stop_rate']:.1%}")
print(f"Average stop margin: {analysis['stop_margin_stats']['avg']:.3f}")
print(f"Average confidence: {analysis['confidence_stats']['avg']:.3f}")
```

## Threshold Optimization

```python
from agents.multi_agent_people_hybrid import recommend_thresholds

# Get recommendations based on historical data
historical_results = [result1, result2, ...]  # Your historical data
tau_s, tau_v = recommend_thresholds(
    historical_results, 
    target_early_stop_rate=0.3  # Target 30% early stop rate
)
```

## Output Format

Results include additional metadata for early stopping analysis:

```python
{
    "mode": "hybrid_mcq_adaptive",
    "thresholds": {"tau_s": 0.5, "tau_v": 0.7},
    "continuation_decisions": [
        {
            "round": "opening",
            "decision": "continue",
            "stop_margin": -0.8,
            "confidence": 0.65,
            "terminated_early": False,
            "tau_s": 0.5,
            "tau_v": 0.7,
            # ... other fields
        }
    ],
    "early_termination_count": 1,
    # ... other fields
}
```

## Key Features

1. **Dual Threshold Control**: Prevents both premature and excessive stopping
2. **Configurable Parameters**: Adjust thresholds based on your requirements
3. **Comprehensive Analysis**: Detailed metrics for performance evaluation
4. **Backward Compatible**: Works with existing debate frameworks
5. **Optimization Tools**: Built-in threshold recommendation system

## Example Scenarios

### High-Stakes Decision Making
- Use conservative thresholds (tau_s=1.0, tau_v=0.8)
- Prioritize accuracy over speed
- Accept longer processing time for thorough analysis

### Real-Time Applications
- Use aggressive thresholds (tau_s=0.0, tau_v=0.6)
- Prioritize speed over exhaustive analysis
- Accept some accuracy trade-off for faster results

### Batch Processing
- Use balanced thresholds (tau_s=0.5, tau_v=0.7)
- Optimize for overall efficiency
- Monitor performance and adjust based on results

## References

- Based on RADAR paper adaptive early stopping approach
- Implements dual-threshold termination controller
- Balances reasoning thoroughness with computational efficiency
