from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    framework: str
    model: str
    temperature: float
    agents_involved: list[str]


@dataclass
class Outcome:
    status: str
    success_score: float


@dataclass
class StepRecord:
    step_index: int
    step_type: str
    agent_id: str
    input: Any
    output: Any
    tool_name: str | None = None
    retrieved_docs: Any | None = None
    is_root_cause: bool = False
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentFaultRecord:
    trajectory_id: str
    task_id: str
    agent_config: AgentConfig
    outcome: Outcome
    fault_injected: bool
    fault_type: str | None
    origin_step: int | None
    injection_params: dict[str, Any] | None
    num_steps: int
    num_agents_involved: int
    source: str
    split: str
    steps: list[StepRecord] = field(default_factory=list)
