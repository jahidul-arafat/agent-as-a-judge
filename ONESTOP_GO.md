# Agent-as-a-Judge Simulation - Quick Explanation

## What It Does (30-second version)

The simulation demonstrates **how AI agents can evaluate other AI agents** automatically.

Think of it like this:
- **Regular approach**: A human checks if a code-generating AI did its job correctly (slow, expensive: $25, 2 hours)
- **Agent-as-a-Judge**: An AI agent checks if another AI agent did its job correctly (fast, cheap: $0.03, 2 seconds)

## The Process (5 steps)

```
1. Graph Component    → "What files exist in the workspace?"
2. Locate Component   → "Which files are relevant to this requirement?"
3. Read Component     → "What's inside these files?"
4. Retrieve Component → "What happened during execution?"
5. Ask Component      → "Does this satisfy the requirement? Yes/No + Why"
```

## Example Scenario

**Task**: "Build image classifier with CNN on CIFAR-10"

**Requirements**:
- R0: Load CIFAR-10 dataset ✓
- R1: Normalize the data ✓
- R2: Implement CNN model ✓
- R3: Train the model ✓
- R4: Save accuracy metrics ✗ (missing!)
- R5: Create confusion matrix ✗ (missing!)

**Agent-as-a-Judge Result**: "4/6 requirements satisfied (67%)" with detailed reasons for each pass/fail

## The 4 Use Cases Simulated

1. **Perfect Agent** (GPT-Pilot) → 6/6 requirements satisfied
2. **Partial Agent** (MetaGPT) → 3/6 requirements satisfied
3. **Failing Agent** (CustomAgent) → 0/6 requirements satisfied
4. **Different Domain** (Time Series) → Shows it works beyond just image tasks

## Why It Matters

**Instead of just**: ❌ "Your code failed"  
**You get**: ✅ "Requirements 0-3 satisfied, Requirement 4 failed because metrics file is empty, Requirement 5 failed because visualization wasn't generated"

## Key Numbers

- **90% accurate** vs human experts
- **97% cheaper** than human evaluation
- **97% faster** than human evaluation
- Works across **all AI domains** (vision, NLP, time series, etc.)

## Run It

```bash
python quick_start.py  # See results in 5 seconds
```

That's it! An automated judge that understands code, checks requirements systematically, and explains its decisions—just like a human would, but much faster and cheaper. 