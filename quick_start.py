"""
QUICK START GUIDE
=================
Agent-as-a-Judge System Simulation

Follow these steps to run the simulation:
"""

# STEP 1: Extract files (if you downloaded the tar.gz)
# tar -xzf agent_judge_simulation.tar.gz

# STEP 2: Run the full interactive simulation
# python run_simulation.py

# STEP 3: Or run individual use cases in Python

from run_simulation import *

print("Agent-as-a-Judge Simulation - Quick Examples")
print("=" * 60)

# Example 1: Quick evaluation without interaction
print("\n1. Running Use Case 1 (High-Performing Agent)...")
task1 = create_sample_task_1()
agent_output1 = create_sample_agent_output_good(task1, "GPT-Pilot")
judge1 = AgentAsAJudge(verbose=False, use_trajectory=True)
result1 = judge1.evaluate_task(task1, agent_output1)
print(f"   Result: {result1['requirements_satisfied']}/{result1['requirements_total']} "
      f"({result1['satisfaction_rate']:.1f}%) requirements satisfied")

# Example 2: Different agent
print("\n2. Running Use Case 2 (Partial Implementation)...")
task2 = create_sample_task_1()
agent_output2 = create_sample_agent_output_partial(task2, "MetaGPT")
judge2 = AgentAsAJudge(verbose=False, use_trajectory=True)
result2 = judge2.evaluate_task(task2, agent_output2)
print(f"   Result: {result2['requirements_satisfied']}/{result2['requirements_total']} "
      f"({result2['satisfaction_rate']:.1f}%) requirements satisfied")

# Example 3: Poor performance
print("\n3. Running Use Case 3 (Low-Performing Agent)...")
task3 = create_sample_task_1()
agent_output3 = create_sample_agent_output_poor(task3, "CustomAgent")
judge3 = AgentAsAJudge(verbose=False, use_trajectory=True)
result3 = judge3.evaluate_task(task3, agent_output3)
print(f"   Result: {result3['requirements_satisfied']}/{result3['requirements_total']} "
      f"({result3['satisfaction_rate']:.1f}%) requirements satisfied")

# Example 4: Different domain
print("\n4. Running Use Case 4 (Time Series Task)...")
task4 = create_sample_task_2()
agent_output4 = create_agent_output_for_task2_good()
judge4 = AgentAsAJudge(verbose=False, use_trajectory=True)
result4 = judge4.evaluate_task(task4, agent_output4)
print(f"   Result: {result4['requirements_satisfied']}/{result4['requirements_total']} "
      f"({result4['satisfaction_rate']:.1f}%) requirements satisfied")

print("\n" + "=" * 60)
print("COMPARISON:")
print("-" * 60)
results = [result1, result2, result3, result4]
for i, result in enumerate(results, 1):
    print(f"Use Case {i}: {result['satisfaction_rate']:5.1f}% satisfied "
          f"(${result['statistics']['total_cost']:.4f}, "
          f"{result['statistics']['total_time']:.2f}s)")

print("=" * 60)
print("\nKey Insights:")
print("• Agent-as-a-Judge evaluates at requirement level, not just pass/fail")
print("• Costs ~$0.001-0.005 per requirement (vs ~$4 for humans)")
print("• Provides detailed justifications for each judgment")
print("• Handles dependencies between requirements")
print("• Generalizes across different AI task domains")

print("\nFor full verbose output, run: python run_simulation.py")
