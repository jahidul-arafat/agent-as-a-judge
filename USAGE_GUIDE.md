# Agent-as-a-Judge Simulation - Complete Usage Guide

## 📦 What's Included

This simulation package contains:

1. **agent_judge_simulation.py** (22 KB)
   - Core framework with 5 components
   - Data structures (Task, Requirement, AgentOutput)
   - Main AgentAsAJudge evaluation system

2. **run_simulation.py** (21 KB)
   - 4 complete use cases
   - Interactive simulation runner
   - Comparison and analysis functions

3. **quick_start.py** (3 KB)
   - Quick examples without interaction
   - Demonstrates programmatic usage

4. **SIMULATION_README.md** (10 KB)
   - Detailed architecture documentation
   - Extension guide

5. **USAGE_GUIDE.md** (this file)
   - Step-by-step instructions

## 🚀 Running the Simulation

### Option 1: Full Interactive Experience (Recommended)

```bash
python run_simulation.py
```

**What happens:**
- Welcome screen with overview
- 4 use cases run sequentially with user prompts
- Detailed verbose output for each evaluation
- Summary comparison at the end
- Takes ~5-10 minutes with reading time

**Best for:** Understanding how the system works in detail

### Option 2: Quick Non-Interactive Demo

```bash
python quick_start.py
```

**What happens:**
- Runs all 4 use cases automatically
- Minimal output (results only)
- Completes in ~5 seconds

**Best for:** Quick verification or API usage examples

### Option 3: Custom Usage in Your Code

```python
from agent_judge_simulation import *

# Create a task
task = Task(
    id=1,
    name="Your Task",
    query="Task description",
    requirements=[...],
    preferences=[...],
    tags=[...]
)

# Create agent output
agent_output = AgentOutput(
    agent_name="YourAgent",
    task_id=1,
    workspace_files=[...],
    trajectory=[...],
    execution_time=100.0,
    cost=2.0,
    completed=True
)

# Evaluate
judge = AgentAsAJudge(verbose=True, use_trajectory=True)
result = judge.evaluate_task(task, agent_output)

# Print results
judge.print_summary_table(result)
```

## 📊 Understanding the 4 Use Cases

### Use Case 1: High-Performing Agent ⭐⭐⭐⭐⭐
```
Agent: GPT-Pilot
Task: Image Classification with CNN
Expected Outcome: 6/6 requirements satisfied (100%)
```

**What it demonstrates:**
- ✅ Complete workspace with all required files
- ✅ Proper implementation of data loading, model, training
- ✅ All metrics and visualizations generated
- ✅ Clean trajectory with no errors
- ✅ All dependencies satisfied in order

**Key Output:**
```
[10:30:45] ✓ [SUCCESS] R0: CIFAR-10 dataset loaded - SATISFIED
[10:30:46] ✓ [SUCCESS] R1: Data normalization implemented - SATISFIED
[10:30:47] ✓ [SUCCESS] R2: CNN model implemented - SATISFIED
...
Overall: 6/6 (100.0%) requirements satisfied
```

### Use Case 2: Partially Complete Agent ⭐⭐⭐
```
Agent: MetaGPT
Task: Same image classification
Expected Outcome: 3/6 requirements satisfied (50%)
```

**What it demonstrates:**
- ✅ Basic implementation exists
- ❌ Missing data normalization
- ❌ Simple model instead of CNN
- ❌ Visualization not generated
- ⚠️ Warnings in trajectory
- 🔗 Dependency checking prevents downstream evaluation

**Key Output:**
```
[10:31:23] ✓ [SUCCESS] R0: Dataset loaded - SATISFIED
[10:31:24] ✗ [ERROR] R1: Normalization missing - UNSATISFIED
[10:31:24] ⚠ [WARN] Skipping R3 due to unsatisfied dependencies
...
Overall: 3/6 (50.0%) requirements satisfied
```

### Use Case 3: Low-Performing Agent ⭐
```
Agent: CustomAgent
Task: Same image classification
Expected Outcome: 0-1/6 requirements satisfied (0-17%)
```

**What it demonstrates:**
- ❌ Minimal workspace (only placeholders)
- ❌ Multiple errors in trajectory
- ❌ Timeout before completion
- ❌ No meaningful files generated
- 🔍 Shows how system handles failures

**Key Output:**
```
[10:32:15] ✗ [ERROR] R0: No dataset found - UNSATISFIED
[10:32:16] ⚠ [WARN] ERROR detected in trajectory: Connection timeout
[10:32:16] ⚠ [WARN] Skipping R1 due to unsatisfied dependencies
...
Overall: 0/6 (0.0%) requirements satisfied
```

### Use Case 4: Different Domain (Time Series) 🔄
```
Agent: OpenHands
Task: LSTM-based Sales Forecasting
Expected Outcome: 5/5 requirements satisfied (100%)
```

**What it demonstrates:**
- 🌐 Framework generalizes to different AI domains
- 📊 Different requirement categories (HCI, visualization)
- 📈 Time series specific evaluation
- 🎯 Interactive HTML report generation

**Key Output:**
```
[10:33:01] ✓ [SUCCESS] R0: Sales dataset loaded - SATISFIED
[10:33:02] ✓ [SUCCESS] R1: Train/test split - SATISFIED
[10:33:03] ✓ [SUCCESS] R2: LSTM model implemented - SATISFIED
[10:33:04] ✓ [SUCCESS] R3: Forecast visualization - SATISFIED
[10:33:05] ✓ [SUCCESS] R4: Interactive HTML report - SATISFIED
...
Overall: 5/5 (100.0%) requirements satisfied
```

## 🔍 Verbose Output Explained

When running with `verbose=True`, you see:

```
[10:30:45] ℹ [INFO] Task Query: Build an image classification...
[10:30:45] ℹ [INFO] Total Requirements: 6
[10:30:45] ℹ [INFO] Agent: GPT-Pilot
[10:30:45] ℹ [INFO] Workspace Files: 5

--------------------------------------------------------------------------------
  Step 1: Building Workspace Graph
--------------------------------------------------------------------------------
  [10:30:45] [GRAPH] INFO: Building workspace graph from 5 files...
  [10:30:45] [GRAPH] INFO:   Mapped: src/data_loader.py (python, 18 lines)
  [10:30:45] [GRAPH] INFO:   Mapped: src/model.py (python, 22 lines)
  ...
  [10:30:45] [GRAPH] INFO: Graph built: 5 files, 3 directories, 3 file types

--------------------------------------------------------------------------------
  Evaluating Requirement R0
--------------------------------------------------------------------------------
  [10:30:45] ℹ [INFO] Criteria: CIFAR-10 dataset is loaded in src/data_loader.py
  [10:30:45] ℹ [INFO] Category: Dataset or Environment
  [10:30:45] ℹ [INFO] Dependencies: None
  
  [10:30:45] [LOCATE] INFO: Locating files for requirement...
  [10:30:45] [LOCATE] INFO:   ✓ Found explicitly mentioned: src/data_loader.py
  [10:30:45] [LOCATE] INFO: Located 1 relevant file(s)
  
  [10:30:45] [READ] INFO: Reading 1 file(s)...
  [10:30:45] [READ] INFO:   ✓ Read src/data_loader.py: 456 chars, 18 lines
  [10:30:45] [READ] INFO:     Preview: import torch from torchvision...
  
  [10:30:45] [RETRIEVE] INFO: Retrieving trajectory data...
  [10:30:45] [RETRIEVE] INFO:   ✓ Step 0: Create src/data_loader.py...
  [10:30:45] [RETRIEVE] INFO: Retrieved 1 relevant step(s)
  
  [10:30:45] [ASK] INFO: Making judgment for: CIFAR-10 dataset...
  [10:30:45] [ASK] INFO: Evidence available:
  [10:30:45] [ASK] INFO:   - File contents: 1 files
  [10:30:45] [ASK] INFO:   - Trajectory steps: 1 steps
  [10:30:45] [ASK] INFO:   - Located files: 1 files
  [10:30:45] [ASK] SUCCESS:   ✓ JUDGMENT: SATISFIED
  [10:30:45] [ASK] INFO:   Justification: Requirement satisfied. Found 1 relevant...
```

## 📈 Performance Metrics

The simulation tracks:

```python
{
    "total_requirements": 6,
    "satisfied": 5,
    "unsatisfied": 1,
    "total_cost": 0.006,      # In USD
    "total_time": 0.60        # In seconds
}
```

**Cost Breakdown:**
- ~$0.001 per requirement evaluation
- ~$0.006 for typical 6-requirement task
- Compare to ~$25 for human evaluation

**Time Breakdown:**
- ~0.1s per requirement (with simulated API calls)
- ~0.6s for typical 6-requirement task
- Compare to ~2 hours for human evaluation

## 🎯 What Each Component Does

### 1. Graph Component
```python
graph = {
    "files": [
        {"path": "src/model.py", "type": "python", "size": 22}
    ],
    "directories": ["src", "results", "results/figures"],
    "file_types": {
        "python": ["src/model.py", "src/train.py"],
        "text": ["results/metrics.txt"]
    }
}
```

### 2. Locate Component
```python
# Input: Requirement about "src/model.py"
# Output: ["src/model.py"]

# Input: Requirement about "visualization"
# Output: ["results/figures/confusion_matrix.png"]
```

### 3. Read Component
```python
# Input: ["src/model.py", "results/metrics.txt"]
# Output: {
#     "src/model.py": "import torch.nn as nn...",
#     "results/metrics.txt": "Test Accuracy: 72.34%..."
# }
```

### 4. Retrieve Component
```python
# Input: Requirement about "accuracy"
# Output: [
#     {"step": 5, "action": "Calculate accuracy", "error": None},
#     {"step": 6, "action": "Save metrics", "error": None}
# ]
```

### 5. Ask Component
```python
# Input: All evidence from above components
# Output: (RequirementStatus.SATISFIED, "Requirement satisfied because...")
```

## 🔧 Customization Examples

### Add Your Own Task

```python
def create_my_task():
    return Task(
        id=100,
        name="My Custom Task",
        query="Your task description here",
        requirements=[
            Requirement(
                id="0",
                criteria="First requirement",
                dependencies=[],
                category="Dataset or Environment"
            ),
            Requirement(
                id="1",
                criteria="Second requirement that depends on first",
                dependencies=["0"],
                category="Machine Learning Method"
            )
        ],
        preferences=["Optional nice-to-have"],
        tags=["CustomTag"]
    )
```

### Add Your Own Agent Output

```python
def create_my_agent_output(task):
    return AgentOutput(
        agent_name="MyAgent",
        task_id=task.id,
        workspace_files=[
            WorkspaceFile(
                path="my_file.py",
                content="print('Hello')",
                file_type="python",
                size_lines=1
            )
        ],
        trajectory=[
            {"step": 0, "action": "Started", "thought": "Beginning", "error": None}
        ],
        execution_time=100.0,
        cost=2.0,
        completed=True
    )
```

### Evaluate Custom Task

```python
task = create_my_task()
output = create_my_agent_output(task)

judge = AgentAsAJudge(verbose=True, use_trajectory=True)
result = judge.evaluate_task(task, output)
judge.print_summary_table(result)
```

## 📚 Key Concepts

### Hierarchical Requirements
Requirements form a DAG (Directed Acyclic Graph):
```
     R0 (Load Data)
    /              \
   R1               R2
(Preprocess)    (Model)
   \              /
        R3
    (Training)
       |
       R4
    (Metrics)
```

If R0 fails, R1 and downstream requirements are skipped.

### Black-Box vs. Gray-Box
- **Black-Box**: Only workspace files (no trajectory)
  - `judge = AgentAsAJudge(use_trajectory=False)`
- **Gray-Box**: Workspace files + trajectory
  - `judge = AgentAsAJudge(use_trajectory=True)`

Gray-box provides richer insights (e.g., error detection).

### Evidence Collection
For each requirement, the judge collects:
1. **Structural**: Which files exist
2. **Content**: What's in the files
3. **Execution**: What happened during generation
4. **Dependencies**: Status of prerequisite requirements

## 🎓 Learning Outcomes

After running this simulation, you'll understand:

1. ✅ How Agent-as-a-Judge evaluates agentic systems
2. ✅ Why component-based design matters
3. ✅ How dependency checking works
4. ✅ The value of gray-box evaluation
5. ✅ Cost/time tradeoffs vs. human evaluation
6. ✅ How to extend the framework
7. ✅ Limitations and failure modes

## 🐛 Troubleshooting

### "Module not found" error
```bash
# Make sure you're in the correct directory
ls agent_judge_simulation.py  # Should see the file
python3 agent_judge_simulation.py  # Should load without errors
```

### Want less verbose output?
```python
judge = AgentAsAJudge(verbose=False)  # Quiet mode
```

### Want to see component calls?
```python
# After evaluation:
print(f"Graph called: {judge.graph_component.call_count} times")
print(f"Locate called: {judge.locate_component.call_count} times")
```

## 📞 Support

This is an educational simulation based on:
- **Paper**: "Agent-as-a-Judge: Evaluate Agents with Agents"
- **Authors**: Zhuge et al. (Meta AI & KAUST)
- **Conference**: ICML 2025
- **Code**: github.com/metauto-ai/agent-as-a-judge

For paper-specific questions: mingchen.zhuge@kaust.edu.sa

## 🎉 Quick Commands Cheat Sheet

```bash
# Full interactive experience
python run_simulation.py

# Quick demo (no interaction)
python quick_start.py

# Test that imports work
python -c "from agent_judge_simulation import *; print('✓ Working!')"

# Run custom evaluation
python
>>> from agent_judge_simulation import *
>>> task = create_sample_task_1()
>>> # Your custom code here...
```

Enjoy exploring Agent-as-a-Judge! 🚀
