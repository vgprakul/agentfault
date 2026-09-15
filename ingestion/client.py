import os
import requests


def send_trajectory(recorder):
    url = os.getenv(
        "INGESTION_URL",
        "http://localhost:8000/trajectories",
    )

    events = [event.to_dict() for event in recorder.events]

    payload = {
        "trajectory_id": recorder.trajectory_id,
        "events": events,
    }

    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()

    return response.json()
