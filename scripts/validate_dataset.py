import json
from collections import Counter
from pathlib import Path


BASELINE_DIR = Path("data/trajectories/baseline")
INJECTED_DIR = Path("data/trajectories/injected")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_record(data, expected_injected):
    errors = []

    required = {
        "trajectory_id",
        "task_id",
        "agent_config",
        "outcome",
        "fault_injected",
        "fault_type",
        "origin_step",
        "injection_params",
        "num_steps",
        "num_agents_involved",
        "source",
        "split",
        "steps",
    }

    missing = required - data.keys()

    if missing:
        errors.append(
            f"missing fields: {sorted(missing)}"
        )
        return errors

    steps = data["steps"]

    if not isinstance(steps, list):
        errors.append("steps is not a list")
        return errors

    if data["num_steps"] != len(steps):
        errors.append(
            f"num_steps={data['num_steps']} "
            f"but len(steps)={len(steps)}"
        )

    step_indices = [
        step.get("step_index")
        for step in steps
    ]

    if len(step_indices) != len(set(step_indices)):
        errors.append("duplicate step_index")

    for step in steps:
        step_required = {
            "step_index",
            "step_type",
            "agent_id",
            "input",
            "output",
            "tool_name",
            "retrieved_docs",
            "is_root_cause",
        }

        missing_step = step_required - step.keys()

        if missing_step:
            errors.append(
                "step missing fields: "
                f"{sorted(missing_step)}"
            )

    if expected_injected:
        if data["fault_injected"] is not True:
            errors.append(
                "fault_injected is not true"
            )

        if data["source"] != "INJECTED":
            errors.append(
                f"source={data['source']}"
            )

        if data["fault_type"] is None:
            errors.append(
                "fault_type is null"
            )

        if data["origin_step"] is None:
            errors.append(
                "origin_step is null"
            )

        if data["injection_params"] is None:
            errors.append(
                "injection_params is null"
            )

        outcome = data["outcome"]

        if outcome.get("status") != "FAIL":
            errors.append(
                f"outcome.status={outcome.get('status')}"
            )

        if outcome.get("success_score") != 0.0:
            errors.append(
                "success_score is not 0.0"
            )

        root_steps = [
            step
            for step in steps
            if step.get("is_root_cause") is True
        ]

        if len(root_steps) != 1:
            errors.append(
                f"expected 1 root cause, "
                f"found {len(root_steps)}"
            )
        else:
            root_index = root_steps[0]["step_index"]

            if (
                root_index != data["origin_step"]
                and data["fault_type"]
                != "PLAN_MISSING_STEP"
            ):
                errors.append(
                    f"root cause step {root_index} "
                    f"!= origin_step "
                    f"{data['origin_step']}"
                )

            if root_index not in step_indices:
                errors.append(
                    "root cause step does not exist"
                )

    else:
        if data["fault_injected"] is not False:
            errors.append(
                "baseline fault_injected is not false"
            )

        if data["fault_type"] is not None:
            errors.append(
                "baseline fault_type is not null"
            )

        if data["origin_step"] is not None:
            errors.append(
                "baseline origin_step is not null"
            )

        if data["source"] != "NATURAL":
            errors.append(
                f"baseline source={data['source']}"
            )

    return errors


def validate_directory(
    directory,
    expected_injected,
):
    files = sorted(
        directory.glob("*.json")
        if expected_injected
        else directory.glob("*.jsonl")
    )

    total = 0
    valid = 0
    invalid = 0
    errors_by_file = {}

    trajectory_ids = set()
    duplicate_ids = []

    fault_counts = Counter()

    for path in files:
        total += 1

        try:
            data = load_json(path)
        except Exception as e:
            invalid += 1
            errors_by_file[path.name] = [
                f"invalid JSON: {e}"
            ]
            continue

        trajectory_id = data.get(
            "trajectory_id"
        )

        if trajectory_id in trajectory_ids:
            duplicate_ids.append(
                trajectory_id
            )

        trajectory_ids.add(trajectory_id)

        errors = validate_record(
            data,
            expected_injected,
        )

        if errors:
            invalid += 1
            errors_by_file[path.name] = errors
        else:
            valid += 1

        if expected_injected:
            fault_counts[
                data.get("fault_type")
            ] += 1

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "duplicate_ids": duplicate_ids,
        "errors": errors_by_file,
        "fault_counts": fault_counts,
    }


def main():
    print("=== BASELINES ===")

    baseline = validate_directory(
        BASELINE_DIR,
        expected_injected=False,
    )

    print(
        f"Files:   {baseline['total']}"
    )
    print(
        f"Valid:   {baseline['valid']}"
    )
    print(
        f"Invalid: {baseline['invalid']}"
    )

    if baseline["duplicate_ids"]:
        print(
            f"Duplicate IDs: "
            f"{len(baseline['duplicate_ids'])}"
        )

    print()
    print("=== INJECTED ===")

    injected = validate_directory(
        INJECTED_DIR,
        expected_injected=True,
    )

    print(
        f"Files:   {injected['total']}"
    )
    print(
        f"Valid:   {injected['valid']}"
    )
    print(
        f"Invalid: {injected['invalid']}"
    )

    if injected["duplicate_ids"]:
        print(
            f"Duplicate IDs: "
            f"{len(injected['duplicate_ids'])}"
        )

    print()
    print("=== FAULT DISTRIBUTION ===")

    for fault, count in sorted(
        injected["fault_counts"].items()
    ):
        print(
            f"{fault}: {count}"
        )

    if injected["errors"]:
        print()
        print("=== ERRORS ===")

        for filename, errors in list(
            injected["errors"].items()
        )[:20]:
            print()
            print(filename)

            for error in errors:
                print(f"  - {error}")

        remaining = (
            len(injected["errors"]) - 20
        )

        if remaining > 0:
            print()
            print(
                f"... and {remaining} more files"
            )

    print()

    if (
        baseline["invalid"] == 0
        and injected["invalid"] == 0
        and not baseline["duplicate_ids"]
        and not injected["duplicate_ids"]
    ):
        print("DATASET VALIDATION PASSED")
    else:
        print("DATASET VALIDATION FAILED")


if __name__ == "__main__":
    main()
