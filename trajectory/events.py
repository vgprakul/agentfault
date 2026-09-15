from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json


@dataclass
class TrajectoryEvent:
    trajectory_id: str
    timestamp: str
    event_type: str
    step: int
    agent: str | None = None
    input: Any = None
    output: Any = None
    tool: str | None = None
    status: str = "success"
    parent_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "step": self.step,
            "agent": self.agent,
            "input": self.input,
            "output": self.output,
            "tool": self.tool,
            "status": self.status,
            "parent_step": self.parent_step,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()

class TrajectoryRecorder:
    def __init__(self, trajectory_id: str):
        self.trajectory_id = trajectory_id
        self.events: list[TrajectoryEvent] = []
        self._step = 0

    def record(
        self,
        event_type: str,
        *,
        agent: str | None = None,
        input: Any = None,
        output: Any = None,
        tool: str | None = None,
        status: str = "success",
        parent_step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryEvent:

        event = TrajectoryEvent(
            trajectory_id=self.trajectory_id,
            timestamp=now(),
            event_type=event_type,
            step=self._step,
            agent=agent,
            input=input,
            output=output,
            tool=tool,
            status=status,
            parent_step=parent_step,
            metadata=metadata or {},
        )

        self.events.append(event)
        self._step += 1

        return event

    def to_jsonl(self) -> str:
        return "\n".join(event.to_json() for event in self.events)
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_jsonl())
            f.write("\n")
