"""
Agent-as-a-Judge Simulation Runner
===================================
Runs 4 different use cases demonstrating the system
"""

from agent_judge_simulation import *
import random


def create_sample_agent_output_good(task: Task, agent_name: str) -> AgentOutput:
    """Create a GOOD agent output (most requirements satisfied)"""
    
    workspace_files = [
        WorkspaceFile(
            path="src/data_loader.py",
            content="""
import torch
from torchvision import datasets, transforms

def load_cifar10():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    return trainset, testset

if __name__ == "__main__":
    train, test = load_cifar10()
    print(f"Loaded {len(train)} training samples, {len(test)} test samples")
""",
            file_type="python",
            size_lines=18
        ),
        WorkspaceFile(
            path="src/model.py",
            content="""
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
""",
            file_type="python",
            size_lines=22
        ),
        WorkspaceFile(
            path="src/train.py",
            content="""
import torch
import torch.optim as optim
from model import CNNModel
from data_loader import load_cifar10

def train_model(epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    
    trainset, testset = load_cifar10()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}')
    
    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    # Save metrics
    with open('results/metrics.txt', 'w') as f:
        f.write(f'Test Accuracy: {accuracy:.2f}%\\n')
        f.write(f'Correct: {correct}/{total}\\n')
    
    print(f'Accuracy: {accuracy:.2f}%')
    return model, accuracy

if __name__ == "__main__":
    train_model()
""",
            file_type="python",
            size_lines=53
        ),
        WorkspaceFile(
            path="results/metrics.txt",
            content="Test Accuracy: 72.34%\nCorrect: 7234/10000\nEpochs: 10\nFinal Loss: 0.823",
            file_type="text",
            size_lines=4
        ),
        WorkspaceFile(
            path="results/figures/confusion_matrix.png",
            content="[Binary PNG data - 1024x768 confusion matrix visualization]",
            file_type="image",
            size_lines=1
        )
    ]
    
    trajectory = [
        {"step": 0, "thought": "Need to load CIFAR-10 dataset", "action": "Create src/data_loader.py", "error": None},
        {"step": 1, "thought": "Implement data normalization", "action": "Add normalization transforms", "error": None},
        {"step": 2, "thought": "Design CNN architecture", "action": "Create src/model.py with CNN", "error": None},
        {"step": 3, "thought": "Implement training loop", "action": "Create src/train.py", "error": None},
        {"step": 4, "thought": "Train model for 10 epochs", "action": "Execute training", "error": None},
        {"step": 5, "thought": "Evaluate on test set", "action": "Calculate accuracy", "error": None},
        {"step": 6, "thought": "Save metrics to file", "action": "Write results/metrics.txt", "error": None},
        {"step": 7, "thought": "Generate confusion matrix", "action": "Create visualization", "error": None},
        {"step": 8, "thought": "Save confusion matrix", "action": "Save to results/figures/confusion_matrix.png", "error": None},
        {"step": 9, "thought": "Task completed successfully", "action": "Finish", "error": None}
    ]
    
    return AgentOutput(
        agent_name=agent_name,
        task_id=task.id,
        workspace_files=workspace_files,
        execution_time=452.3,
        cost=3.45,
        trajectory=trajectory,
        completed=True
    )


def create_sample_agent_output_partial(task: Task, agent_name: str) -> AgentOutput:
    """Create a PARTIAL agent output (some requirements missing)"""
    
    workspace_files = [
        WorkspaceFile(
            path="src/data_loader.py",
            content="""
# Simple data loader - no normalization implemented yet
import torch
from torchvision import datasets

def load_data():
    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)
    return trainset, testset
""",
            file_type="python",
            size_lines=8
        ),
        WorkspaceFile(
            path="src/model.py",
            content="""
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3072, 10)
    
    def forward(self, x):
        x = x.view(-1, 3072)
        return self.fc(x)
""",
            file_type="python",
            size_lines=10
        ),
        WorkspaceFile(
            path="results/metrics.txt",
            content="Accuracy: 45.2%",
            file_type="text",
            size_lines=1
        )
    ]
    
    trajectory = [
        {"step": 0, "thought": "Loading dataset", "action": "Create data_loader.py", "error": None},
        {"step": 1, "thought": "Creating simple model", "action": "Create model.py", "error": None},
        {"step": 2, "thought": "Training model", "action": "Run training", "error": "WARNING: No normalization applied"},
        {"step": 3, "thought": "Saving metrics", "action": "Write metrics.txt", "error": None},
        {"step": 4, "thought": "Need to create confusion matrix", "action": "Attempting visualization", "error": "ERROR: matplotlib not found"},
        {"step": 5, "thought": "Time limit approaching", "action": "Stopping early", "error": None}
    ]
    
    return AgentOutput(
        agent_name=agent_name,
        task_id=task.id,
        workspace_files=workspace_files,
        execution_time=180.5,
        cost=1.89,
        trajectory=trajectory,
        completed=False
    )


def create_sample_agent_output_poor(task: Task, agent_name: str) -> AgentOutput:
    """Create a POOR agent output (most requirements not satisfied)"""
    
    workspace_files = [
        WorkspaceFile(
            path="main.py",
            content="""
# TODO: Implement CIFAR-10 classification
print("Starting implementation...")
""",
            file_type="python",
            size_lines=3
        ),
        WorkspaceFile(
            path="README.md",
            content="# CIFAR-10 Classification\n\nWork in progress...",
            file_type="markdown",
            size_lines=3
        )
    ]
    
    trajectory = [
        {"step": 0, "thought": "Need to understand CIFAR-10", "action": "Research dataset", "error": None},
        {"step": 1, "thought": "Creating project structure", "action": "Create main.py", "error": None},
        {"step": 2, "thought": "Should I use PyTorch or TensorFlow?", "action": "Thinking...", "error": None},
        {"step": 3, "thought": "Attempting to download dataset", "action": "pip install torchvision", "error": "ERROR: Connection timeout"},
        {"step": 4, "thought": "Cannot proceed without dataset", "action": "Creating placeholder", "error": None},
        {"step": 5, "thought": "Time limit reached", "action": "Stopping", "error": "TIMEOUT"}
    ]
    
    return AgentOutput(
        agent_name=agent_name,
        task_id=task.id,
        workspace_files=workspace_files,
        execution_time=1800.0,
        cost=0.95,
        trajectory=trajectory,
        completed=False
    )


def create_sample_task_2() -> Task:
    """Create sample task 2: Time series forecasting"""
    return Task(
        id=2,
        name="Time Series Forecasting with LSTM",
        query="Develop a sales forecasting system using LSTM on store sales data. "
               "Load data in src/data_loader.py, implement LSTM in src/model.py, "
               "save forecast plot to results/figures/forecast.png, and create "
               "interactive HTML report at results/report.html.",
        requirements=[
            Requirement(
                id="0",
                criteria="Sales dataset is loaded in src/data_loader.py",
                dependencies=[],
                category="Dataset or Environment"
            ),
            Requirement(
                id="1",
                criteria="Time series data is split into train/test sets in src/data_loader.py",
                dependencies=["0"],
                category="Data preprocessing and postprocessing"
            ),
            Requirement(
                id="2",
                criteria="LSTM model is implemented in src/model.py",
                dependencies=[],
                category="Machine Learning Method"
            ),
            Requirement(
                id="3",
                criteria="Forecast results are plotted and saved as results/figures/forecast.png",
                dependencies=["1", "2"],
                category="Visualization"
            ),
            Requirement(
                id="4",
                criteria="HTML report with interactive elements is generated at results/report.html",
                dependencies=["3"],
                category="Human Computer Interaction"
            )
        ],
        preferences=[
            "Model should capture seasonal trends",
            "HTML report should allow exploring different time horizons"
        ],
        tags=["Time Series", "LSTM", "Forecasting"]
    )


def create_agent_output_for_task2_good() -> AgentOutput:
    """Good output for task 2"""
    workspace_files = [
        WorkspaceFile(
            path="src/data_loader.py",
            content="import pandas as pd\nimport numpy as np\n\ndef load_sales_data():\n    data = pd.read_csv('sales.csv')\n    train = data[:int(0.8*len(data))]\n    test = data[int(0.8*len(data)):]\n    return train, test",
            file_type="python",
            size_lines=8
        ),
        WorkspaceFile(
            path="src/model.py",
            content="import torch.nn as nn\n\nclass LSTMModel(nn.Module):\n    def __init__(self, input_size, hidden_size, num_layers):\n        super().__init__()\n        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)\n        self.fc = nn.Linear(hidden_size, 1)\n    \n    def forward(self, x):\n        out, _ = self.lstm(x)\n        out = self.fc(out[:, -1, :])\n        return out",
            file_type="python",
            size_lines=12
        ),
        WorkspaceFile(
            path="results/figures/forecast.png",
            content="[PNG image data - forecast visualization with predictions vs actuals]",
            file_type="image",
            size_lines=1
        ),
        WorkspaceFile(
            path="results/report.html",
            content="<!DOCTYPE html><html><head><title>Sales Forecast Report</title></head><body><h1>Interactive Sales Forecast</h1><div id='chart'></div><script src='plotly.js'></script></body></html>",
            file_type="html",
            size_lines=1
        )
    ]
    
    trajectory = [
        {"step": 0, "action": "Load sales dataset", "thought": "Reading CSV file", "error": None},
        {"step": 1, "action": "Split train/test", "thought": "80/20 split", "error": None},
        {"step": 2, "action": "Implement LSTM", "thought": "Create model architecture", "error": None},
        {"step": 3, "action": "Train model", "thought": "Training for 50 epochs", "error": None},
        {"step": 4, "action": "Generate forecast plot", "thought": "Creating visualization", "error": None},
        {"step": 5, "action": "Create HTML report", "thought": "Adding interactive Plotly chart", "error": None}
    ]
    
    return AgentOutput(
        agent_name="OpenHands",
        task_id=2,
        workspace_files=workspace_files,
        execution_time=325.7,
        cost=4.12,
        trajectory=trajectory,
        completed=True
    )


def run_use_case_1():
    """Use Case 1: Perfect Agent (Most Requirements Satisfied)"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "USE CASE 1: HIGH-PERFORMING AGENT" + " "*25 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\nScenario: GPT-Pilot successfully completes an image classification task")
    print("Expected: Most requirements satisfied, good alignment with criteria")
    
    input("\nPress Enter to start evaluation...")
    
    task = create_sample_task_1()
    agent_output = create_sample_agent_output_good(task, "GPT-Pilot")
    
    judge = AgentAsAJudge(verbose=True, use_trajectory=True)
    result = judge.evaluate_task(task, agent_output)
    judge.print_summary_table(result)
    
    return result


def run_use_case_2():
    """Use Case 2: Partial Implementation"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*18 + "USE CASE 2: PARTIALLY COMPLETE AGENT" + " "*22 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\nScenario: MetaGPT completes basic implementation but misses key requirements")
    print("Expected: Some requirements satisfied, missing normalization and visualization")
    
    input("\nPress Enter to start evaluation...")
    
    task = create_sample_task_1()
    agent_output = create_sample_agent_output_partial(task, "MetaGPT")
    
    judge = AgentAsAJudge(verbose=True, use_trajectory=True)
    result = judge.evaluate_task(task, agent_output)
    judge.print_summary_table(result)
    
    return result


def run_use_case_3():
    """Use Case 3: Poor Performance"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "USE CASE 3: LOW-PERFORMING AGENT" + " "*25 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\nScenario: Agent fails to make meaningful progress due to errors and timeout")
    print("Expected: Most/all requirements unsatisfied, trajectory shows errors")
    
    input("\nPress Enter to start evaluation...")
    
    task = create_sample_task_1()
    agent_output = create_sample_agent_output_poor(task, "CustomAgent")
    
    judge = AgentAsAJudge(verbose=True, use_trajectory=True)
    result = judge.evaluate_task(task, agent_output)
    judge.print_summary_table(result)
    
    return result


def run_use_case_4():
    """Use Case 4: Different Task Type (Time Series)"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*15 + "USE CASE 4: DIFFERENT DOMAIN (TIME SERIES)" + " "*18 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\nScenario: OpenHands works on time series forecasting task")
    print("Expected: Demonstrates Agent-as-a-Judge generalization to different AI domains")
    
    input("\nPress Enter to start evaluation...")
    
    task = create_sample_task_2()
    agent_output = create_agent_output_for_task2_good()
    
    judge = AgentAsAJudge(verbose=True, use_trajectory=True)
    result = judge.evaluate_task(task, agent_output)
    judge.print_summary_table(result)
    
    return result


def compare_results(results: List[Dict]):
    """Compare results across all use cases"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "CROSS-CASE COMPARISON" + " "*32 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\n{:<20} {:<15} {:<20} {:<15} {:<15}".format(
        "Use Case", "Agent", "Satisfied/Total", "Success Rate", "Cost"))
    print("-"*80)
    
    for i, result in enumerate(results, 1):
        use_case = f"Use Case {i}"
        agent = result['agent']
        satisfied = f"{result['requirements_satisfied']}/{result['requirements_total']}"
        success_rate = f"{result['satisfaction_rate']:.1f}%"
        cost = f"${result['statistics']['total_cost']:.4f}"
        
        print(f"{use_case:<20} {agent:<15} {satisfied:<20} {success_rate:<15} {cost:<15}")
    
    print("-"*80)
    
    # Calculate averages
    avg_satisfaction = sum(r['satisfaction_rate'] for r in results) / len(results)
    total_cost = sum(r['statistics']['total_cost'] for r in results)
    total_time = sum(r['statistics']['total_time'] for r in results)
    
    print(f"\nAverage Satisfaction Rate: {avg_satisfaction:.1f}%")
    print(f"Total Evaluation Cost: ${total_cost:.4f}")
    print(f"Total Evaluation Time: {total_time:.2f}s")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Agent-as-a-Judge provides granular requirement-level feedback")
    print("2. Dependency tracking ensures logical evaluation order")
    print("3. Trajectory analysis (gray-box) helps identify execution issues")
    print("4. Cost-effective compared to human evaluation (~$0.001-0.005 per requirement)")
    print("5. Framework generalizes across different AI task domains")
    print("="*80)


def main():
    """Main simulation runner"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " "*15 + "AGENT-AS-A-JUDGE SYSTEM SIMULATION" + " "*29 + "║")
    print("║" + " "*20 + "ICML 2025 (Zhuge et al.)" + " "*34 + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    print("\nThis simulation demonstrates the Agent-as-a-Judge framework through 4 use cases:")
    print("  1. High-performing agent (most requirements satisfied)")
    print("  2. Partially complete agent (some requirements missing)")
    print("  3. Low-performing agent (most requirements unsatisfied)")
    print("  4. Different domain (time series forecasting)")
    
    print("\nEach use case shows:")
    print("  • Workspace graph construction")
    print("  • File location and content reading")
    print("  • Trajectory analysis (gray-box evaluation)")
    print("  • Requirement-level judgments with justifications")
    print("  • Dependency checking and evaluation ordering")
    
    input("\nPress Enter to begin simulation...")
    
    results = []
    
    # Run all use cases
    results.append(run_use_case_1())
    input("\n\nPress Enter to continue to next use case...")
    
    results.append(run_use_case_2())
    input("\n\nPress Enter to continue to next use case...")
    
    results.append(run_use_case_3())
    input("\n\nPress Enter to continue to next use case...")
    
    results.append(run_use_case_4())
    input("\n\nPress Enter to see comparison...")
    
    # Compare results
    compare_results(results)
    
    print("\n\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " "*25 + "SIMULATION COMPLETE" + " "*34 + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    print("\nTo learn more:")
    print("  • Paper: ICML 2025")
    print("  • Code: github.com/metauto-ai/agent-as-a-judge")
    print("  • Contact: mingchen.zhuge@kaust.edu.sa")


if __name__ == "__main__":
    main()
