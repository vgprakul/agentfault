from uuid import uuid4
import sys 
from agent.graph import build_graph
from trajectory.events import TrajectoryRecorder
from ingestion.client import send_trajectory


graph = build_graph()
query = " ".join(sys.argv[1:]).strip()

if not query:
    print("Usage: python main.py <query>")
    raise SystemExit(1)

trajectory_id = f"traj_{uuid4().hex[:8]}"
recorder = TrajectoryRecorder(trajectory_id)

recorder.record(
    "trajectory_start",
    input={"query": query},
)

try:
    result = graph.invoke({
        "query": query,
        "retrieved_context": "",
        "tool_result": "",
        "research": "",
        "review": "",
        "answer": "",
        "recorder": recorder,
    })

    recorder.record(
        "trajectory_end",
        output=result["answer"],
    )

    print(result["answer"])

except Exception as e:
    recorder.record(
        "trajectory_end",
        status="error",
        output=str(e),
    )

    print(f"Agent failed: {e}")

finally:
    try:
        recorder.save(f"data/trajectories/{trajectory_id}.jsonl")
        print(f"Saved: data/trajectories/{trajectory_id}.jsonl")
    except Exception as e:
        print(f"Local save failed: {e}")

    try:
        result = send_trajectory(recorder)
        print(f"Ingested: {result}")
    except Exception as e:
        print(f"Ingestion failed: {e}")

    print(f"\nTrajectory: {trajectory_id}")
