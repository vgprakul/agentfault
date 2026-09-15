# AgentFault Injector

The AgentFault injector generates synthetic agent trajectories and injects
controlled faults into them according to the AgentFault taxonomy.

The injector is responsible for:

1. Generating baseline agent trajectories.
2. Selecting a fault type.
3. Selecting an applicable trajectory step.
4. Applying a deterministic fault operator.
5. Marking the injected step as the root cause.
6. Recording injection metadata.
7. Producing dataset-ready `AgentFaultRecord` objects.

## Fault taxonomy

The injector currently defines 20 fault types.

### Planning

| Fault | Description |
|---|---|
| `PLAN_MISSING_STEP` | Remove a required step from the agent's plan or trajectory. |
| `PLAN_INCORRECT_SUCCESS_CRITERIA` | Modify the success criteria used by the agent. |
| `PLAN_INVALID_DEPENDENCY` | Introduce an invalid dependency between planned steps. |

### Tool usage

| Fault | Description |
|---|---|
| `TOOL_WRONG_TOOL` | Replace the tool selected for a tool call. |
| `TOOL_WRONG_ARGUMENT` | Replace or corrupt an argument passed to a tool. |
| `TOOL_UNNECESSARY_CALL` | Insert a tool call that is not required for the task. |
| `TOOL_FAILED_RECOVERY` | Cause the agent's recovery behaviour after a tool failure to fail. |

### Knowledge / retrieval

| Fault | Description |
|---|---|
| `KNOW_RETRIEVAL_FAILURE` | Cause retrieval to return missing, incorrect, or unusable information. |
| `KNOW_CITATION_MISMATCH` | Make the cited source inconsistent with the information used. |
| `KNOW_CONTEXT_TRUNCATION` | Remove part of the context available to the agent. |

### Multi-agent

| Fault | Description |
|---|---|
| `MA_INCORRECT_HANDOFF` | Route a task or result to the wrong agent. |
| `MA_INFORMATION_LOSS` | Remove information during communication between agents. |
| `MA_MISSING_RESPONSIBILITY` | Leave a required responsibility unassigned. |
| `MA_ROLE_OVERLAP` | Give multiple agents overlapping responsibility for the same task. |

### Control

| Fault | Description |
|---|---|
| `CTRL_LOOP` | Cause the agent to repeatedly execute the same or equivalent action. |
| `CTRL_PREMATURE_TERMINATION` | Terminate execution before the task is complete. |
| `CTRL_EXCESSIVE_EXPLORATION` | Cause the agent to perform unnecessary exploration before completing the task. |

### Security

| Fault | Description |
|---|---|
| `SEC_PROMPT_INJECTION` | Inject adversarial instructions into agent-visible input or context. |
| `SEC_UNAUTHORIZED_ACTION` | Cause the agent to perform an action outside its authorization. |
| `SEC_CROSS_USER_DATA_LEAKAGE` | Introduce information belonging to another user into the trajectory. |

## Injection contract

Every injected trajectory must preserve the original dataset schema.

An injected trajectory must:

- have `fault_injected=true`
- contain the corresponding `fault_type`
- record the affected `origin_step`
- record the operator parameters in `injection_params`
- mark the root-cause step using `is_root_cause=true`
- use `source="INJECTED"`
- preserve deterministic behaviour when a random seed is provided
- receive a unique trajectory ID

Successful baseline trajectories use:

```text
fault_injected = false
fault_type = null
origin_step = null
injection_params = null
source = NATURAL
