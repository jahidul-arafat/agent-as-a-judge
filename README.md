# Agent-as-a-Judge System Simulation

A comprehensive Python simulation of the Agent-as-a-Judge framework from the ICML 2025 paper by Zhuge et al. (Meta AI & KAUST).

## Overview

This simulation demonstrates how Agent-as-a-Judge evaluates code-generating agentic systems through:
- **5 Core Components**: Graph, Locate, Read, Retrieve, Ask
- **Hierarchical Requirements**: DAG structure with dependencies
- **Gray-box Evaluation**: Using trajectory data for deeper insights
- **Multiple Use Cases**: Different agent performance levels and task domains

## Quick Start

```bash
python run_simulation.py
```

The simulation will walk you through 4 use cases interactively.

## Architecture

### Core Components

1. **GraphComponent**: Builds workspace structure graph
   - Maps all files and directories
   - Categorizes by file type
   - Creates dependency relationships

2. **LocateComponent**: Finds relevant files for each requirement
   - Parses requirement criteria
   - Matches files by path, name, and category
   - Intelligently focuses evaluation

3. **ReadComponent**: Reads file contents
   - Supports 33+ file formats
   - Extracts multimodal data
   - Provides content previews

4. **RetrieveComponent**: Analyzes execution trajectory
   - Extracts relevant steps
   - Identifies errors and warnings
   - Focuses on recent actions

5. **AskComponent**: Makes final judgment
   - Synthesizes all evidence
   - Returns SATISFIED/UNSATISFIED
   - Provides detailed justification

### Data Structures

```python
@dataclass
class Requirement:
    id: str
    criteria: str
    dependencies: List[str]
    category: str
    status: RequirementStatus
    justification: str

@dataclass
class Task:
    id: int
    name: str
    query: str
    requirements: List[Requirement]
    preferences: List[str]
    tags: List[str]

@dataclass
class AgentOutput:
    agent_name: str
    workspace_files: List[WorkspaceFile]
    trajectory: List[Dict]
    execution_time: float
    cost: float
```

## Use Cases

### Use Case 1: High-Performing Agent
**Scenario**: GPT-Pilot successfully completes image classification task
- **Agent**: GPT-Pilot
- **Task**: Image classification with CNN on CIFAR-10
- **Expected**: 5-6/6 requirements satisfied
- **Features Demonstrated**:
  - Complete workspace with all required files
  - Proper data normalization
  - Model implementation and training
  - Metrics and visualization generation
  - Clean trajectory with no errors

### Use Case 2: Partially Complete Agent
**Scenario**: MetaGPT implements basic solution but misses key requirements
- **Agent**: MetaGPT
- **Task**: Same image classification task
- **Expected**: 2-3/6 requirements satisfied
- **Features Demonstrated**:
  - Missing data preprocessing (normalization)
  - Simple model instead of CNN
  - No visualization generated
  - Trajectory shows warnings
  - Dependency checking prevents evaluation of downstream requirements

### Use Case 3: Low-Performing Agent
**Scenario**: CustomAgent fails due to errors and timeout
- **Agent**: CustomAgent
- **Task**: Same image classification task
- **Expected**: 0-1/6 requirements satisfied
- **Features Demonstrated**:
  - Minimal workspace (only placeholder files)
  - Multiple errors in trajectory
  - Timeout before completion
  - Shows how Agent-as-a-Judge handles failures

### Use Case 4: Different Domain
**Scenario**: OpenHands works on time series forecasting
- **Agent**: OpenHands
- **Task**: LSTM-based sales forecasting
- **Expected**: 4-5/5 requirements satisfied
- **Features Demonstrated**:
  - Framework generalizes to different AI domains
  - Interactive HTML report generation
  - Time series specific requirements
  - HCI component evaluation

## Key Features Demonstrated

### 1. Hierarchical Requirements with Dependencies
```
R0: Load dataset (no dependencies)
R1: Preprocess data (depends on R0)
R2: Implement model (no dependencies)
R3: Train model (depends on R1, R2)
R4: Save metrics (depends on R3)
R5: Create visualization (depends on R3, R4)
```

If R1 fails, R3 cannot be evaluated (dependency check).

### 2. Verbose Logging System
Every operation is logged with:
- Timestamp
- Component name
- Operation details
- Results and findings
- Errors and warnings

Example output:
```
[10:30:45] ✓ [SUCCESS] Agent-as-a-Judge Evaluation: Task 1
  [10:30:45] [GRAPH] INFO: Building workspace graph from 5 files...
  [10:30:45] [GRAPH] INFO:   Mapped: src/data_loader.py (python, 18 lines)
  [10:30:45] [LOCATE] INFO: Locating files for requirement...
  [10:30:45] [LOCATE] INFO:   ✓ Found explicitly mentioned: src/data_loader.py
```

### 3. Evidence Collection
For each requirement, the system collects:
- Located relevant files
- File contents
- Trajectory steps mentioning the requirement
- Workspace structure information

### 4. Judgment Process
The Ask component:
1. Reviews all collected evidence
2. Checks for file existence
3. Verifies content is non-empty
4. Looks for errors in trajectory
5. Makes binary decision (SATISFIED/UNSATISFIED)
6. Provides detailed justification

### 5. Statistics Tracking
```python
evaluation_stats = {
    "total_requirements": 6,
    "satisfied": 5,
    "unsatisfied": 1,
    "total_cost": 0.006,  # $0.006
    "total_time": 1.234   # seconds
}
```

## Comparison with Human Evaluation

| Metric | Human-as-a-Judge | Agent-as-a-Judge |
|--------|------------------|------------------|
| Time per task | ~2 hours | ~2 seconds |
| Cost per task | ~$25 | ~$0.03 |
| Requirements evaluated | 6 | 6 |
| Provides justification | Yes | Yes |
| Scalable | No | Yes |
| Consistent | Variable | High |

## Understanding the Output

### Summary Table
```
EVALUATION SUMMARY
================================================================================
Task: Image Classification with CNN (ID: 1)
Agent: GPT-Pilot

Requirement ID    Category                  Status        Time (s)  
--------------------------------------------------------------------------------
R0                Dataset or Environment    ✓ SATISFIED   0.234
R1                Data preprocessing        ✓ SATISFIED   0.189
R2                Machine Learning Method   ✓ SATISFIED   0.145
R3                Machine Learning Method   ✓ SATISFIED   0.201
R4                Performance Metrics       ✓ SATISFIED   0.156
R5                Visualization             ✗ UNSATISFIED 0.178
--------------------------------------------------------------------------------

Overall: 5/6 (83.3%) requirements satisfied
Total Time: 1.10s
Total Cost: $0.0060
```

### Component Performance Gains
As shown in the paper:
- Ask only: 65% alignment
- + Graph: 76% alignment (+11%)
- + Read: 82% alignment (+6%)
- + Locate: 90% alignment (+8%)
- + Retrieve: 90% alignment (varies)

## Extending the Simulation

### Adding New Tasks
```python
def create_custom_task() -> Task:
    return Task(
        id=3,
        name="Your Task Name",
        query="Task description...",
        requirements=[
            Requirement(
                id="0",
                criteria="Your requirement",
                dependencies=[],
                category="Dataset or Environment"
            ),
            # Add more requirements...
        ],
        preferences=["Optional preference"],
        tags=["Tag1", "Tag2"]
    )
```

### Adding New Agents
```python
def create_custom_agent_output(task: Task) -> AgentOutput:
    return AgentOutput(
        agent_name="YourAgent",
        task_id=task.id,
        workspace_files=[...],  # Your generated files
        trajectory=[...],        # Execution steps
        execution_time=300.0,
        cost=2.50,
        completed=True
    )
```

### Customizing Components
Each component can be customized by extending the base class:

```python
class CustomLocateComponent(AgentAsAJudgeComponent):
    def __init__(self, verbose: bool = True):
        super().__init__("CUSTOM_LOCATE", verbose)
    
    def execute(self, requirement, workspace_graph):
        # Your custom logic
        return located_files
```

## Files in This Simulation

1. **agent_judge_simulation.py**: Core framework implementation
   - All 5 components
   - Data structures
   - Main AgentAsAJudge class

2. **run_simulation.py**: Use case runner
   - 4 complete use cases
   - Sample task and agent output generators
   - Comparison and analysis functions

3. **README.md**: This file

## Key Insights from Simulation

1. **Granular Feedback**: Unlike binary pass/fail, Agent-as-a-Judge shows exactly which requirements succeeded/failed

2. **Dependency Awareness**: Requirements with unsatisfied dependencies are properly skipped

3. **Trajectory Value**: Gray-box evaluation (with trajectory) catches issues that black-box evaluation misses

4. **Cost-Effective**: ~$0.001 per requirement vs. ~$4 per requirement for human evaluation

5. **Generalization**: Same framework works across different AI domains (vision, NLP, time series)

6. **Interpretability**: Detailed justifications explain every judgment

7. **Scalability**: Can evaluate hundreds of agent runs per day

## Limitations Demonstrated

1. **Synthetic Data Detection**: May be fooled by well-named fake datasets
2. **Semantic Nuances**: Might miss difference between "setting" vs. "tuning" parameters
3. **Complex Validation**: Cannot verify correctness of ML algorithms without execution
4. **Context Limits**: Very large workspaces might exceed LLM context windows

## Paper Citation

```bibtex
@inproceedings{zhuge2025agent,
  title={Agent-as-a-Judge: Evaluate Agents with Agents},
  author={Zhuge, Mingchen and Zhao, Changsheng and Ashley, Dylan R and others},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

## Contact

For questions about this simulation:
- Based on work by: Meta AI & KAUST
- Paper contacts: mingchen.zhuge@kaust.edu.sa, cszhao@meta.com

## License

This is an educational simulation based on published research. 
Original paper: ICML 2025
Code availability: github.com/metauto-ai/agent-as-a-judge
