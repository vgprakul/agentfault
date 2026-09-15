from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="AgentFault Trajectory Ingestion")

TRAJECTORY_DIR = Path("./data/trajectories")
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


class Trajectory(BaseModel):
    trajectory_id: str
    events: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/trajectories")
def ingest_trajectory(trajectory: Trajectory):
    path = TRAJECTORY_DIR / f"{trajectory.trajectory_id}.json"

    try:
        path.write_text(trajectory.model_dump_json(indent=2))
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "accepted",
        "trajectory_id": trajectory.trajectory_id,
    }
