import json
from pathlib import Path

from trajectory.events import TrajectoryEvent


def load_trajectory(path: str | Path) -> list[TrajectoryEvent]:
    events = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(TrajectoryEvent(**json.loads(line)))

    return events


def save_trajectory(events: list[TrajectoryEvent], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps({
                "trajectory_id": event.trajectory_id,
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "step": event.step,
                "agent": event.agent,
                "input": event.input,
                "output": event.output,
                "tool": event.tool,
                "status": event.status,
                "parent_step": event.parent_step,
                "metadata": event.metadata,
            }) + "\n")
