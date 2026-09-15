from pathlib import Path

from injector.compatibility import compatible_faults
from injector.real_trajectory import (
    events_to_record,
    load_events,
)


def main():
    baseline_dir = Path("data/trajectories/baseline")

    total = 0

    for path in sorted(baseline_dir.glob("*.jsonl")):
        events = load_events(path)
        record = events_to_record(events)

        compatible = compatible_faults(record)

        print(f"\n{path.name}")
        print(f"  steps: {len(record.steps)}")
        print(
            "  faults:",
            ", ".join(
                fault.value
                for fault in compatible
            ),
        )

        total += 1

    print(f"\nChecked {total} trajectories")


if __name__ == "__main__":
    main()
