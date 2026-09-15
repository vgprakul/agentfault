from copy import deepcopy

from injector.faults import Fault
from injector.operators import OPERATORS
from injector.schema import AgentFaultRecord


class FaultInjector:

    def inject(
        self,
        trajectory: AgentFaultRecord,
        fault: Fault,
    ) -> AgentFaultRecord:

        if fault.fault_type not in OPERATORS:
            raise ValueError(
                f"Unsupported fault type: "
                f"{fault.fault_type}"
            )

        injected = deepcopy(trajectory)

        injected.trajectory_id = (
            f"{trajectory.trajectory_id}"
            f"_fault_{fault.fault_type.value}"
            f"_step_{fault.step}"
        )

        injected.fault_injected = True
        injected.fault_type = (
            fault.fault_type.value
        )
        injected.origin_step = fault.step
        injected.source = "INJECTED"

        target = next(
            (
                step
                for step in injected.steps
                if step.step_index == fault.step
            ),
            None,
        )

        if target is None:
            raise ValueError(
                f"Step {fault.step} not found"
            )

        metadata = OPERATORS[
            fault.fault_type
        ](
            injected,
            target,
            fault.parameters,
        )

        injected.injection_params = {
            **metadata,
            "random_seed": fault.seed,
        }

        # Most faults keep the original target step.
        # PLAN_MISSING_STEP removes it, so the nearest
        # surviving step becomes the observable root cause.
        root_step = next(
            (
                step
                for step in injected.steps
                if step.step_index == fault.step
            ),
            None,
        )

        if root_step is None:
            root_step = next(
                (
                    step
                    for step in injected.steps
                    if step.step_index > fault.step
                ),
                None,
            )

        if root_step is None and injected.steps:
            root_step = injected.steps[-1]

        if root_step is not None:
            root_step.is_root_cause = True

        injected.outcome.status = "FAIL"
        injected.outcome.success_score = 0.0

        return injected
