import json
from dataclasses import asdict
from pathlib import Path

from injector.faults import Fault, FaultType
from injector.injector import FaultInjector
from injector.real_trajectory import (
    events_to_record,
    load_events,
)


def main():
    baseline_dir = Path("data/trajectories/baseline")
    injected_dir = Path("data/trajectories/injected")
    injected_dir.mkdir(parents=True, exist_ok=True)

    # Find a real trajectory containing a compatible TOOL_CALL
    trajectory_path = None

    for path in baseline_dir.glob("*.jsonl"):
        events = load_events(path)

        if any(event.event_type == "tool_call" for event in events):
            trajectory_path = path
            break

    if trajectory_path is None:
        raise RuntimeError(
            "No baseline trajectory with tool_call found"
        )

    print(f"Using baseline: {trajectory_path}")

    # Real Gemini trajectory -> AgentFaultRecord
    events = load_events(trajectory_path)
    record = events_to_record(events)

    # Find the first TOOL_CALL
    target = next(
        step
        for step in record.steps
        if step.step_type == "TOOL_CALL"
    )

    print(f"Target step: {target.step_index}")
    print(f"Original input: {target.input}")

    if not isinstance(target.input, dict):
        raise RuntimeError(
            f"Expected dict input, got {type(target.input).__name__}"
        )

    if "service" in target.input:
        argument_name = "service"
        injected_value = "invalid_service"
    elif "expression" in target.input:
        argument_name = "expression"
        injected_value = "invalid_expression"
    else:
        raise RuntimeError(
            f"No supported argument found: {target.input}"
        )

    fault = Fault(
        fault_type=FaultType.TOOL_WRONG_ARGUMENT,
        step=target.step_index,
        parameters={
            "argument_name": argument_name,
            "injected_value": injected_value,
        },
        seed=42,
    )

    # Generic injector
    injected = FaultInjector().inject(record, fault)

    print(f"Injected input: {injected.steps[
        next(
            i
            for i, step in enumerate(injected.steps)
            if step.step_index == target.step_index
        )
    ].input}")

    output_path = (
        injected_dir
        / f"{injected.trajectory_id}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(injected), f, indent=2)

    print(f"Saved: {output_path}")
    print(f"Fault: {injected.fault_type}")
    print(f"Root cause step: {injected.origin_step}")
    print(f"Source: {injected.source}")
    print(f"Outcome: {injected.outcome.status}")
    print(f"Success score: {injected.outcome.success_score}")


if __name__ == "__main__":
    main()
