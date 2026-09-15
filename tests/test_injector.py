from injector.faults import Fault, FaultType
from injector.generator import TrajectoryGenerator
from injector.injector import FaultInjector


generator = TrajectoryGenerator(seed=42)

trajectory = generator.generate(
    trajectory_id="test_trajectory",
    task_id="test_task",
)

fault = Fault(
    fault_type=FaultType.TOOL_WRONG_TOOL,
    step=5,
    parameters={
        "operator": "REPLACE_TOOL",
        "original_tool": "calculator",
        "injected_tool": "wrong_tool",
    },
    seed=42,
)

injector = FaultInjector()

injected = injector.inject(
    trajectory,
    fault,
)

assert injected.trajectory_id == (
    "test_trajectory_fault_TOOL_WRONG_TOOL_step_5"
)

assert injected.fault_injected is True
assert injected.fault_type == "TOOL_WRONG_TOOL"
assert injected.origin_step == 5
assert injected.source == "INJECTED"

step = next(
    step for step in injected.steps
    if step.step_index == 5 
)

assert step.tool_name == "wrong_tool"
assert step.is_root_cause is True

assert injected.injection_params["operator"] == "REPLACE_TOOL"
assert injected.injection_params["original_tool"] == "calculator"
assert injected.injection_params["injected_tool"] == "wrong_tool"
assert injected.injection_params["random_seed"] == 42

assert injected.outcome.status == "FAIL"
assert injected.outcome.success_score == 0.0

print("All injector tests passed.")
