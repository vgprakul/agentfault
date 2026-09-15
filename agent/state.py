from typing import TypedDict

from trajectory.events import TrajectoryRecorder


class AgentState(TypedDict):
    query: str
    retrieved_context: str
    tool_result: str
    research: str
    review: str
    answer: str
    recorder: TrajectoryRecorder
