"""
Agent-as-a-Judge System Simulation
===================================
This module simulates the Agent-as-a-Judge framework for evaluating code-generating agents.

Based on the ICML 2025 paper by Zhuge et al. (Meta AI & KAUST)
"""

import json
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class RequirementStatus(Enum):
    """Status of a requirement evaluation"""
    SATISFIED = "SATISFIED"
    UNSATISFIED = "UNSATISFIED"
    PENDING = "PENDING"


@dataclass
class Requirement:
    """Represents a single requirement in DevAI"""
    id: str
    criteria: str
    dependencies: List[str]
    category: str
    status: RequirementStatus = RequirementStatus.PENDING
    justification: str = ""
    
    def __repr__(self):
        return f"R{self.id}: {self.criteria[:50]}... [{self.status.value}]"


@dataclass
class Task:
    """Represents a complete DevAI task"""
    id: int
    name: str
    query: str
    requirements: List[Requirement]
    preferences: List[str]
    tags: List[str]
    
    def get_requirement_by_id(self, req_id: str) -> Optional[Requirement]:
        """Get requirement by ID"""
        for req in self.requirements:
            if req.id == req_id:
                return req
        return None


@dataclass
class WorkspaceFile:
    """Represents a file in the generated workspace"""
    path: str
    content: str
    file_type: str
    size_lines: int


@dataclass
class AgentOutput:
    """Output from a code-generating agent"""
    agent_name: str
    task_id: int
    workspace_files: List[WorkspaceFile]
    execution_time: float
    cost: float
    trajectory: List[Dict]
    completed: bool


class AgentAsAJudgeComponent:
    """Base class for Agent-as-a-Judge components"""
    
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.call_count = 0
    
    def log(self, message: str, level: str = "INFO"):
        """Log with verbose output"""
        if self.verbose:
            indent = "  " * 2
            timestamp = time.strftime("%H:%M:%S")
            print(f"{indent}[{timestamp}] [{self.name}] {level}: {message}")
    
    def execute(self, *args, **kwargs):
        """Execute component logic"""
        self.call_count += 1
        raise NotImplementedError


class GraphComponent(AgentAsAJudgeComponent):
    """Builds workspace structure graph"""
    
    def __init__(self, verbose: bool = True):
        super().__init__("GRAPH", verbose)
    
    def execute(self, workspace_files: List[WorkspaceFile]) -> Dict:
        """Build dependency graph of workspace"""
        self.log(f"Building workspace graph from {len(workspace_files)} files...")
        
        graph = {
            "files": [],
            "directories": set(),
            "dependencies": {},
            "file_types": {}
        }
        
        for wf in workspace_files:
            # Extract directory
            parts = wf.path.split('/')
            if len(parts) > 1:
                graph["directories"].add('/'.join(parts[:-1]))
            
            # Categorize by type
            if wf.file_type not in graph["file_types"]:
                graph["file_types"][wf.file_type] = []
            graph["file_types"][wf.file_type].append(wf.path)
            
            graph["files"].append({
                "path": wf.path,
                "type": wf.file_type,
                "size": wf.size_lines
            })
            
            self.log(f"  Mapped: {wf.path} ({wf.file_type}, {wf.size_lines} lines)")
        
        graph["directories"] = list(graph["directories"])
        
        self.log(f"Graph built: {len(graph['files'])} files, "
                f"{len(graph['directories'])} directories, "
                f"{len(graph['file_types'])} file types")
        
        return graph


class LocateComponent(AgentAsAJudgeComponent):
    """Locates relevant files for a requirement"""
    
    def __init__(self, verbose: bool = True):
        super().__init__("LOCATE", verbose)
    
    def execute(self, requirement: Requirement, workspace_graph: Dict) -> List[str]:
        """Find files relevant to requirement"""
        self.log(f"Locating files for requirement: {requirement.criteria[:60]}...")
        
        relevant_files = []
        criteria_lower = requirement.criteria.lower()
        
        # Extract file paths mentioned in criteria
        for file_info in workspace_graph["files"]:
            path = file_info["path"]
            
            # Check if path is mentioned in criteria
            if path in requirement.criteria or path.split('/')[-1] in criteria_lower:
                relevant_files.append(path)
                self.log(f"  ✓ Found explicitly mentioned: {path}")
            # Check by category
            elif requirement.category == "Dataset or Environment" and "data" in path.lower():
                relevant_files.append(path)
                self.log(f"  ✓ Found by category match: {path}")
            elif requirement.category == "Machine Learning Method" and "model" in path.lower():
                relevant_files.append(path)
                self.log(f"  ✓ Found by category match: {path}")
            elif requirement.category == "Visualization" and ("figure" in path.lower() or "plot" in path.lower()):
                relevant_files.append(path)
                self.log(f"  ✓ Found by category match: {path}")
        
        if not relevant_files:
            self.log(f"  ⚠ No specific files located, will use all workspace files", "WARN")
            relevant_files = [f["path"] for f in workspace_graph["files"][:3]]  # Limit to 3
        
        self.log(f"Located {len(relevant_files)} relevant file(s)")
        return relevant_files


class ReadComponent(AgentAsAJudgeComponent):
    """Reads file contents"""
    
    def __init__(self, verbose: bool = True):
        super().__init__("READ", verbose)
    
    def execute(self, file_paths: List[str], workspace_files: List[WorkspaceFile]) -> Dict[str, str]:
        """Read contents of specified files"""
        self.log(f"Reading {len(file_paths)} file(s)...")
        
        contents = {}
        
        for path in file_paths:
            # Find file in workspace
            for wf in workspace_files:
                if wf.path == path:
                    contents[path] = wf.content
                    preview = wf.content[:100].replace('\n', ' ')
                    self.log(f"  ✓ Read {path}: {len(wf.content)} chars, "
                           f"{wf.size_lines} lines")
                    self.log(f"    Preview: {preview}...")
                    break
        
        if not contents:
            self.log("  ⚠ No file contents retrieved", "WARN")
        
        return contents


class RetrieveComponent(AgentAsAJudgeComponent):
    """Retrieves relevant trajectory information"""
    
    def __init__(self, verbose: bool = True):
        super().__init__("RETRIEVE", verbose)
    
    def execute(self, requirement: Requirement, trajectory: List[Dict]) -> List[Dict]:
        """Extract relevant steps from trajectory"""
        self.log(f"Retrieving trajectory data for requirement...")
        
        relevant_steps = []
        criteria_keywords = set(requirement.criteria.lower().split())
        
        # Look for steps that mention files or concepts from requirement
        for step in trajectory[-10:]:  # Focus on last 10 steps
            step_text = step.get("action", "").lower() + " " + step.get("thought", "").lower()
            
            # Check for keyword overlap
            step_keywords = set(step_text.split())
            overlap = criteria_keywords.intersection(step_keywords)
            
            if len(overlap) >= 2 or any(keyword in step_text for keyword in ['error', 'success', 'saved', 'created']):
                relevant_steps.append(step)
                self.log(f"  ✓ Step {step['step']}: {step['action'][:60]}...")
                if step.get("error"):
                    self.log(f"    ⚠ Error detected: {step['error'][:50]}", "WARN")
        
        self.log(f"Retrieved {len(relevant_steps)} relevant step(s) from trajectory")
        return relevant_steps


class AskComponent(AgentAsAJudgeComponent):
    """Makes final judgment on requirement"""
    
    def __init__(self, verbose: bool = True):
        super().__init__("ASK", verbose)
        self.llm_call_cost = 0.001  # Simulated cost per call
    
    def execute(self, requirement: Requirement, evidence: Dict) -> Tuple[RequirementStatus, str]:
        """Make judgment based on collected evidence"""
        self.log(f"Making judgment for: {requirement.criteria[:60]}...")
        self.log(f"Evidence available:")
        self.log(f"  - File contents: {len(evidence.get('file_contents', {}))} files")
        self.log(f"  - Trajectory steps: {len(evidence.get('trajectory_steps', []))} steps")
        self.log(f"  - Located files: {len(evidence.get('located_files', []))} files")
        
        # Simulate LLM reasoning
        time.sleep(0.1)  # Simulate API call
        
        # Simple rule-based simulation of judgment
        file_contents = evidence.get('file_contents', {})
        trajectory_steps = evidence.get('trajectory_steps', [])
        located_files = evidence.get('located_files', [])
        
        # Check if files exist
        has_files = len(located_files) > 0
        
        # Check for errors in trajectory
        has_errors = any(step.get('error') for step in trajectory_steps)
        
        # Check if file contents are non-empty
        has_content = any(len(content) > 50 for content in file_contents.values())
        
        # Make decision
        satisfied = has_files and not has_errors and has_content
        
        if satisfied:
            status = RequirementStatus.SATISFIED
            justification = (f"Requirement satisfied. Found {len(located_files)} relevant file(s) "
                           f"with appropriate content. No errors detected in execution trajectory. "
                           f"Files contain {sum(len(c) for c in file_contents.values())} total characters.")
            self.log(f"  ✓ JUDGMENT: SATISFIED", "SUCCESS")
        else:
            status = RequirementStatus.UNSATISFIED
            reasons = []
            if not has_files:
                reasons.append("no relevant files found")
            if has_errors:
                reasons.append("errors in execution")
            if not has_content:
                reasons.append("insufficient file content")
            
            justification = f"Requirement not satisfied. Issues: {', '.join(reasons)}."
            self.log(f"  ✗ JUDGMENT: UNSATISFIED - {', '.join(reasons)}", "ERROR")
        
        self.log(f"  Justification: {justification}")
        
        return status, justification


class AgentAsAJudge:
    """Main Agent-as-a-Judge evaluation system"""
    
    def __init__(self, verbose: bool = True, use_trajectory: bool = True):
        self.verbose = verbose
        self.use_trajectory = use_trajectory
        
        # Initialize components
        self.graph_component = GraphComponent(verbose)
        self.locate_component = LocateComponent(verbose)
        self.read_component = ReadComponent(verbose)
        self.retrieve_component = RetrieveComponent(verbose) if use_trajectory else None
        self.ask_component = AskComponent(verbose)
        
        self.evaluation_stats = {
            "total_requirements": 0,
            "satisfied": 0,
            "unsatisfied": 0,
            "total_cost": 0.0,
            "total_time": 0.0
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Main logger"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            symbol = {
                "INFO": "ℹ",
                "SUCCESS": "✓",
                "ERROR": "✗",
                "WARN": "⚠"
            }.get(level, "•")
            print(f"[{timestamp}] {symbol} [{level}] {message}")
    
    def print_header(self, text: str):
        """Print section header"""
        if self.verbose:
            print("\n" + "="*80)
            print(f"  {text}")
            print("="*80)
    
    def print_subheader(self, text: str):
        """Print subsection header"""
        if self.verbose:
            print("\n" + "-"*80)
            print(f"  {text}")
            print("-"*80)
    
    def evaluate_requirement(self, requirement: Requirement, agent_output: AgentOutput,
                            workspace_graph: Dict) -> Dict:
        """Evaluate a single requirement"""
        start_time = time.time()
        
        self.print_subheader(f"Evaluating Requirement R{requirement.id}")
        self.log(f"Criteria: {requirement.criteria}")
        self.log(f"Category: {requirement.category}")
        self.log(f"Dependencies: {requirement.dependencies if requirement.dependencies else 'None'}")
        
        # Step 1: Locate relevant files
        located_files = self.locate_component.execute(requirement, workspace_graph)
        
        # Step 2: Read file contents
        file_contents = self.read_component.execute(located_files, agent_output.workspace_files)
        
        # Step 3: Retrieve trajectory information (if available)
        trajectory_steps = []
        if self.use_trajectory and self.retrieve_component:
            trajectory_steps = self.retrieve_component.execute(requirement, agent_output.trajectory)
        
        # Step 4: Collect evidence
        evidence = {
            "located_files": located_files,
            "file_contents": file_contents,
            "trajectory_steps": trajectory_steps,
            "workspace_graph": workspace_graph
        }
        
        # Step 5: Make judgment
        status, justification = self.ask_component.execute(requirement, evidence)
        
        elapsed_time = time.time() - start_time
        cost = self.ask_component.llm_call_cost  # Simulated cost
        
        # Update requirement
        requirement.status = status
        requirement.justification = justification
        
        # Update stats
        self.evaluation_stats["total_requirements"] += 1
        if status == RequirementStatus.SATISFIED:
            self.evaluation_stats["satisfied"] += 1
        else:
            self.evaluation_stats["unsatisfied"] += 1
        self.evaluation_stats["total_cost"] += cost
        self.evaluation_stats["total_time"] += elapsed_time
        
        return {
            "requirement_id": requirement.id,
            "status": status.value,
            "justification": justification,
            "time": elapsed_time,
            "cost": cost
        }
    
    def evaluate_task(self, task: Task, agent_output: AgentOutput) -> Dict:
        """Evaluate all requirements in a task"""
        self.print_header(f"Agent-as-a-Judge Evaluation: Task {task.id} - {task.name}")
        
        self.log(f"Task Query: {task.query[:100]}...")
        self.log(f"Total Requirements: {len(task.requirements)}")
        self.log(f"Agent: {agent_output.agent_name}")
        self.log(f"Agent Execution Time: {agent_output.execution_time:.2f}s")
        self.log(f"Agent Cost: ${agent_output.cost:.2f}")
        self.log(f"Workspace Files: {len(agent_output.workspace_files)}")
        
        # Step 1: Build workspace graph
        self.print_subheader("Step 1: Building Workspace Graph")
        workspace_graph = self.graph_component.execute(agent_output.workspace_files)
        
        # Step 2: Evaluate each requirement
        results = []
        for req in task.requirements:
            # Check dependencies
            if req.dependencies:
                self.log(f"Checking dependencies for R{req.id}...")
                all_deps_satisfied = True
                for dep_id in req.dependencies:
                    dep_req = task.get_requirement_by_id(dep_id)
                    if dep_req and dep_req.status != RequirementStatus.SATISFIED:
                        all_deps_satisfied = False
                        self.log(f"  Dependency R{dep_id} not satisfied", "WARN")
                
                if not all_deps_satisfied:
                    self.log(f"Skipping R{req.id} due to unsatisfied dependencies", "WARN")
                    req.status = RequirementStatus.UNSATISFIED
                    req.justification = "Dependencies not satisfied"
                    continue
            
            # Evaluate requirement
            result = self.evaluate_requirement(req, agent_output, workspace_graph)
            results.append(result)
        
        # Calculate final statistics
        satisfied_count = sum(1 for r in task.requirements if r.status == RequirementStatus.SATISFIED)
        total_count = len(task.requirements)
        satisfaction_rate = (satisfied_count / total_count * 100) if total_count > 0 else 0
        
        self.print_header("Evaluation Complete")
        self.log(f"Requirements Satisfied: {satisfied_count}/{total_count} ({satisfaction_rate:.1f}%)", "SUCCESS")
        self.log(f"Total Evaluation Time: {self.evaluation_stats['total_time']:.2f}s")
        self.log(f"Total Evaluation Cost: ${self.evaluation_stats['total_cost']:.4f}")
        
        return {
            "task_id": task.id,
            "task_name": task.name,
            "agent": agent_output.agent_name,
            "requirements_satisfied": satisfied_count,
            "requirements_total": total_count,
            "satisfaction_rate": satisfaction_rate,
            "results": results,
            "statistics": self.evaluation_stats.copy()
        }
    
    def print_summary_table(self, evaluation_result: Dict):
        """Print a summary table of results"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nTask: {evaluation_result['task_name']} (ID: {evaluation_result['task_id']})")
        print(f"Agent: {evaluation_result['agent']}")
        print(f"\n{'Requirement ID':<15} {'Category':<25} {'Status':<12} {'Time (s)':<10}")
        print("-"*80)
        
        for result in evaluation_result['results']:
            req_id = f"R{result['requirement_id']}"
            # Find requirement to get category
            category = "N/A"
            status = result['status']
            time_taken = result['time']
            
            status_symbol = "✓" if status == "SATISFIED" else "✗"
            print(f"{req_id:<15} {category:<25} {status_symbol} {status:<10} {time_taken:<10.3f}")
        
        print("-"*80)
        print(f"\nOverall: {evaluation_result['requirements_satisfied']}/{evaluation_result['requirements_total']} "
              f"({evaluation_result['satisfaction_rate']:.1f}%) requirements satisfied")
        print(f"Total Time: {evaluation_result['statistics']['total_time']:.2f}s")
        print(f"Total Cost: ${evaluation_result['statistics']['total_cost']:.4f}")
        print("="*80)


def create_sample_task_1() -> Task:
    """Create sample task 1: Image classification with CNN"""
    return Task(
        id=1,
        name="Image Classification with CNN",
        query="Build an image classification system using CNN on CIFAR-10 dataset. "
               "Implement data loading in src/data_loader.py, CNN model in src/model.py, "
               "training in src/train.py, and save accuracy metrics to results/metrics.txt. "
               "Create confusion matrix visualization in results/figures/confusion_matrix.png.",
        requirements=[
            Requirement(
                id="0",
                criteria="CIFAR-10 dataset is loaded in src/data_loader.py",
                dependencies=[],
                category="Dataset or Environment"
            ),
            Requirement(
                id="1",
                criteria="Data preprocessing with normalization is implemented in src/data_loader.py",
                dependencies=["0"],
                category="Data preprocessing and postprocessing"
            ),
            Requirement(
                id="2",
                criteria="CNN model is implemented in src/model.py",
                dependencies=[],
                category="Machine Learning Method"
            ),
            Requirement(
                id="3",
                criteria="Model training is implemented in src/train.py",
                dependencies=["1", "2"],
                category="Machine Learning Method"
            ),
            Requirement(
                id="4",
                criteria="Accuracy metrics are saved in results/metrics.txt",
                dependencies=["3"],
                category="Performance Metrics"
            ),
            Requirement(
                id="5",
                criteria="Confusion matrix is generated and saved as results/figures/confusion_matrix.png",
                dependencies=["3", "4"],
                category="Visualization"
            )
        ],
        preferences=[
            "Model should achieve >70% accuracy on test set",
            "Training should complete in reasonable time"
        ],
        tags=["Computer Vision", "Supervised Learning", "CNN"]
    )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  AGENT-AS-A-JUDGE SYSTEM SIMULATION")
    print("  Based on ICML 2025 Paper (Zhuge et al.)")
    print("="*80)
    
    # Create sample task
    task = create_sample_task_1()
    
    print(f"\nTask Created: {task.name}")
    print(f"Requirements: {len(task.requirements)}")
    print(f"Tags: {', '.join(task.tags)}")
