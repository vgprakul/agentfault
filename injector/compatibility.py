from injector.faults import FaultType
from injector.schema import AgentFaultRecord


# Faults that require a specific step type.
FAULT_STEP_TYPES = {
    FaultType.PLAN_MISSING_STEP: {
        "LLM_CALL",
    },

    FaultType.PLAN_INCORRECT_SUCCESS_CRITERIA: {
        "LLM_CALL",
    },

    FaultType.PLAN_INVALID_DEPENDENCY: {
        "LLM_CALL",
    },

    FaultType.TOOL_WRONG_TOOL: {
        "TOOL_CALL",
    },

    FaultType.TOOL_WRONG_ARGUMENT: {
        "TOOL_CALL",
    },

    FaultType.TOOL_UNNECESSARY_CALL: {
        "TOOL_CALL",
    },

    FaultType.TOOL_FAILED_RECOVERY: {
        "TOOL_CALL",
    },

    FaultType.KNOW_RETRIEVAL_FAILURE: {
        "RETRIEVAL",
    },

    FaultType.KNOW_CITATION_MISMATCH: {
        "LLM_CALL",
    },

    FaultType.KNOW_CONTEXT_TRUNCATION: {
        "LLM_CALL",
    },

    FaultType.MA_INCORRECT_HANDOFF: {
        "HANDOFF",
    },

    FaultType.MA_INFORMATION_LOSS: {
        "HANDOFF",
        "LLM_CALL",
    },

    FaultType.MA_MISSING_RESPONSIBILITY: {
        "HANDOFF",
    },

    FaultType.MA_ROLE_OVERLAP: {
        "HANDOFF",
    },

    FaultType.CTRL_LOOP: {
        "LLM_CALL",
        "TOOL_CALL",
    },

    FaultType.CTRL_PREMATURE_TERMINATION: {
        "LLM_CALL",
    },

    FaultType.CTRL_EXCESSIVE_EXPLORATION: {
        "LLM_CALL",
    },

    FaultType.SEC_PROMPT_INJECTION: {
        "LLM_CALL",
    },

    FaultType.SEC_UNAUTHORIZED_ACTION: {
        "TOOL_CALL",
    },

    FaultType.SEC_CROSS_USER_DATA_LEAKAGE: {
        "LLM_CALL",
    },
}


def _step_shape_compatible(
    fault_type: FaultType,
    step,
) -> bool:
    """
    Check whether the actual captured step contains
    the structure required by the fault operator.
    """

    if fault_type == FaultType.PLAN_INVALID_DEPENDENCY:
        return isinstance(step.input, dict)

    if fault_type == FaultType.TOOL_WRONG_ARGUMENT:
        return (
            isinstance(step.input, dict)
            and bool(step.input)
        )

    if fault_type == FaultType.TOOL_WRONG_TOOL:
        return step.tool_name is not None

    if fault_type == FaultType.TOOL_UNNECESSARY_CALL:
        return True

    if fault_type == FaultType.TOOL_FAILED_RECOVERY:
        return True

    if fault_type == FaultType.KNOW_RETRIEVAL_FAILURE:
        return True

    if fault_type == FaultType.KNOW_CITATION_MISMATCH:
        return True

    if fault_type == FaultType.KNOW_CONTEXT_TRUNCATION:
        return isinstance(
            step.input,
            (dict, str),
        )

    if fault_type == FaultType.MA_INCORRECT_HANDOFF:
        return True

    if fault_type == FaultType.MA_INFORMATION_LOSS:
        return isinstance(
            step.input,
            (dict, str),
        )

    if fault_type == FaultType.MA_MISSING_RESPONSIBILITY:
        return True

    if fault_type == FaultType.MA_ROLE_OVERLAP:
        return True

    if fault_type == FaultType.CTRL_LOOP:
        return True

    if fault_type == FaultType.CTRL_PREMATURE_TERMINATION:
        return True

    if fault_type == FaultType.CTRL_EXCESSIVE_EXPLORATION:
        return True

    if fault_type == FaultType.SEC_PROMPT_INJECTION:
        return isinstance(
            step.input,
            (dict, str),
        )

    if fault_type == FaultType.SEC_UNAUTHORIZED_ACTION:
        return step.tool_name is not None

    if fault_type == FaultType.SEC_CROSS_USER_DATA_LEAKAGE:
        return True

    return True


def compatible_steps(
    record: AgentFaultRecord,
    fault_type: FaultType,
):
    """
    Return actual steps from the trajectory that can
    safely receive this fault.
    """

    allowed_types = FAULT_STEP_TYPES.get(
        fault_type,
        set(),
    )

    return [
        step
        for step in record.steps
        if (
            step.step_type in allowed_types
            and _step_shape_compatible(
                fault_type,
                step,
            )
        )
    ]


def compatible_faults(
    record: AgentFaultRecord,
):
    """
    Return all faults that are structurally compatible
    with this trajectory, along with their step indices.
    """

    result = {}

    for fault_type in FaultType:
        steps = compatible_steps(
            record,
            fault_type,
        )

        if steps:
            result[fault_type] = [
                step.step_index
                for step in steps
            ]

    return result
