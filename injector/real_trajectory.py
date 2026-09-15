import json
from pathlib import Path

from injector.schema import (
    AgentConfig,
    AgentFaultRecord,
    Outcome,
    StepRecord,
)
from trajectory.events import TrajectoryEvent


EVENT_TO_STEP_TYPE = {
    "llm_call": "LLM_CALL",
    "tool_call": "TOOL_CALL",
    "retrieval": "RETRIEVAL",
    "handoff": "HANDOFF",
}


def load_events(path: str | Path) -> list[TrajectoryEvent]:
    events = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(
                    TrajectoryEvent(**json.loads(line))
                )

    return events


def events_to_record(
    events: list[TrajectoryEvent],
) -> AgentFaultRecord:

    if not events:
        raise ValueError("Trajectory contains no events")

    trajectory_id = events[0].trajectory_id

    start_event = next(
        (
            event
            for event in events
            if event.event_type == "trajectory_start"
        ),
        None,
    )

    task_id = trajectory_id

    if start_event and start_event.input:
        task_id = str(start_event.input)

    agents = sorted(
        {
            event.agent
            for event in events
            if event.agent is not None
        }
    )

    steps = []

    for event in events:
        step_type = EVENT_TO_STEP_TYPE.get(event.event_type)

        if step_type is None:
            continue

        steps.append(
            StepRecord(
                step_index=event.step,
                step_type=step_type,
                agent_id=event.agent or "unknown",
                input=event.input,
                output=event.output,
                tool_name=event.tool,
                retrieved_docs=(
                    event.output
                    if step_type == "RETRIEVAL"
                    else None
                ),
                status=event.status,
                metadata=event.metadata or {},
            )
        )

    return AgentFaultRecord(
        trajectory_id=trajectory_id,
        task_id=task_id,
        agent_config=AgentConfig(
            framework="LangGraph",
            model="gemini-3.6-flash",
            temperature=0.0,
            agents_involved=agents,
        ),
        outcome=Outcome(
            status="SUCCESS",
            success_score=1.0,
        ),
        fault_injected=False,
        fault_type=None,
        origin_step=None,
        injection_params=None,
        num_steps=len(steps),
        num_agents_involved=len(agents),
        source="NATURAL",
        split="TRAIN",
        steps=steps,
    )
