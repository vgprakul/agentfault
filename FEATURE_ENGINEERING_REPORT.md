# Feature engineering implementation report

Implemented directly in the current AgentFault repository. The workspace initially contained only an empty Git directory, so the supplied archive was unpacked into this root. All original archive files remain byte-for-byte unchanged. No classifiers or workflow redesigns were added.

## Run

```powershell
python -m pip install -r requirements-features.txt
python scripts/build_features.py --input dataset/agentfault-100.jsonl
python scripts/build_features.py --input data/trajectories --output data/features/real
python scripts/compare_features.py
python -m unittest tests.test_feature_engineering -v
```

## Results

- Main dataset: 100 trajectories processed, 100 trajectory rows, 815 step rows, 100 root-positive rows, zero skipped records.
- Main dataset labels: 0 clean, 100 faulty. No clean samples were fabricated.
- Real recordings: 3 clean trajectories, 3 trajectory rows, 21 step rows, zero skipped records.
- Main grouped split: 70 train tasks, 15 validation tasks, 15 test tasks, seed 42. All variants sharing a task stay together; the test suite verifies this with multiple variants per group.
- The three real recordings share one task group and therefore all stay in training; no three-way split is possible for one group.
- Embeddings: actual pretrained sentence-transformers/all-MiniLM-L6-v2, cached by model and text. No trained classifiers.
- Label/provenance exclusion and prefix invariance tests pass. Normalized step position is retrospective metadata, excluded from predictive features.

## Tests

14 feature-engineering tests pass, including CSV counts, duplicate rejection, malformed records, root labeling, ordered indices, empty trajectories, semantic ranges, missing statuses, matching separate tool results, loops, grouped splits, provenance invariance, future invariance, and train-only preprocessing.

Full discovery: 14 passing tests and 2 pre-existing import errors in tests/test_injector.py and tests/test_pipeline.py. Both import TrajectoryGenerator from injector.generator, which does not define it. Original generator/tests were not altered.

## Created files and final added structure

```text
feature_engineering/
    __init__.py
    common.py
    loader.py
    trajectory_features.py
    step_features.py
    semantic_features.py
    preprocessing.py
    splitting.py
    pipeline.py
    README.md
scripts/
    build_features.py
    compare_features.py
tests/
    test_feature_engineering.py
requirements-features.txt
FEATURE_ENGINEERING_REPORT.md
data/features/
    trajectory_features.csv
    step_features.csv
    split_manifest.csv
    feature_dictionary.json
    build_report.json
    .cache/                     # Cached embeddings and model weights
    real/                       # Equivalent outputs for three clean recordings
```

Modified original files: none. Existing agent/, injector/, trajectory/, dataset/, ingestion/, k8s/, and other scripts/tests remain in place. The existing .gitignore excludes data/, so generated outputs are available locally but ignored by Git.

## Actual extracted comparison

These are different tasks and sources, so this is an extraction demonstration, not a controlled clean/faulty experiment.

| Feature | Clean | Tool fault | Control fault |
|---|---:|---:|---:|
| total_steps | 7 | 8 | 10 |
| total_tool_calls | 1 | 1 | 3 |
| tool_error_count | 0 | 0 | 0 |
| repeated_action_count | 0 | 0 | 2 |
| loop_indicator | 0 | 0 | 1 |
| avg_semantic_drift | 0.6025 | 0.6289 | 0.6481 |
| max_semantic_drift | 0.9403 | 0.8229 | 0.8229 |

Selected trajectories:

- Clean: `traj_19bc05a7` (NONE)
- Tool fault: `AF-0024_fault_TOOL_WRONG_TOOL_step_5` (TOOL_WRONG_TOOL)
- Control fault: `AF-0035_fault_CTRL_LOOP_step_5` (CTRL_LOOP)

### Actual root-positive step CSV row

```json
{
  "trajectory_id": "AF-0024_fault_TOOL_WRONG_TOOL_step_5",
  "task_id": "TASK-0024",
  "step_index": "5",
  "normalized_step_position": "0.625",
  "is_root_cause": "1",
  "step_type": "TOOL_CALL",
  "agent_id": "researcher",
  "tool_name": "wrong_tool",
  "status": "UNKNOWN",
  "has_error": "0",
  "is_tool_step": "1",
  "is_retrieval_step": "0",
  "is_llm_step": "0",
  "is_handoff_step": "0",
  "input_length_chars": "28",
  "output_length_chars": "13",
  "input_length_words": "2",
  "output_length_words": "1",
  "cumulative_tool_calls": "1",
  "cumulative_retrievals": "1",
  "cumulative_errors": "0",
  "cumulative_retries": "0",
  "cumulative_handoffs": "1",
  "cumulative_unique_tools": "1",
  "cumulative_unique_agents": "2",
  "repeated_action_count_so_far": "0",
  "previous_step_same_type": "0",
  "previous_step_same_agent": "1",
  "previous_step_same_tool": "0",
  "steps_since_last_successful_action": "",
  "task_step_similarity": "0.2753128739711029",
  "semantic_drift": "0.7246871260288972",
  "previous_step_similarity": "0.15312344644194123"
}
```

## All trajectory predictive feature columns (77)

- `total_steps`
- `total_agents`
- `total_tool_calls`
- `total_tool_responses`
- `total_retrieval_steps`
- `total_llm_steps`
- `total_handoffs`
- `total_retries`
- `total_errors`
- `unique_tools_used`
- `unique_agents`
- `tool_calls_per_step`
- `retrievals_per_step`
- `retries_per_step`
- `errors_per_step`
- `handoffs_per_step`
- `tool_call_count`
- `successful_tool_call_count`
- `failed_tool_call_count`
- `tool_success_rate`
- `tool_error_count`
- `duplicate_tool_call_count`
- `repeated_same_arguments_count`
- `consecutive_tool_failures_max`
- `tool_result_missing_count`
- `unique_tool_count`
- `tool_diversity`
- `average_tool_latency_ms`
- `max_tool_latency_ms`
- `repeated_action_count`
- `repeated_tool_argument_count`
- `consecutive_identical_action_max`
- `retry_count`
- `loop_indicator`
- `premature_termination_indicator`
- `max_repeated_sequence_length`
- `steps_since_last_successful_action`
- `retrieval_count`
- `empty_retrieval_count`
- `retrieval_failure_count`
- `total_retrieved_text_length`
- `average_retrieved_text_length`
- `max_retrieved_text_length`
- `average_retrieval_confidence`
- `minimum_retrieval_confidence`
- `agent_count`
- `unique_agent_count`
- `agent_switch_count`
- `handoff_count`
- `dominant_agent`
- `dominant_agent_step_count`
- `sender_receiver_similarity`
- `handoff_information_loss_score`
- `total_input_chars`
- `total_output_chars`
- `total_input_words`
- `total_output_words`
- `average_step_input_length`
- `average_step_output_length`
- `max_step_output_length`
- `total_retrieved_chars`
- `average_retrieved_chars`
- `success_step_count`
- `failed_step_count`
- `error_step_count`
- `failure_step_ratio`
- `first_error_step_position`
- `avg_task_step_similarity`
- `min_task_step_similarity`
- `final_task_step_similarity`
- `avg_semantic_drift`
- `max_semantic_drift`
- `final_semantic_drift`
- `semantic_drift_slope`
- `avg_consecutive_step_similarity`
- `min_consecutive_step_similarity`
- `max_semantic_change`

Labels: `fault_injected`, `fault_type`, `outcome_status`, `origin_step`.

Metadata: `trajectory_id`, `task_id`, `source`.

## All step predictive feature columns (28)

- `step_type`
- `agent_id`
- `tool_name`
- `status`
- `has_error`
- `is_tool_step`
- `is_retrieval_step`
- `is_llm_step`
- `is_handoff_step`
- `input_length_chars`
- `output_length_chars`
- `input_length_words`
- `output_length_words`
- `cumulative_tool_calls`
- `cumulative_retrievals`
- `cumulative_errors`
- `cumulative_retries`
- `cumulative_handoffs`
- `cumulative_unique_tools`
- `cumulative_unique_agents`
- `repeated_action_count_so_far`
- `previous_step_same_type`
- `previous_step_same_agent`
- `previous_step_same_tool`
- `steps_since_last_successful_action`
- `task_step_similarity`
- `semantic_drift`
- `previous_step_similarity`

Labels: `is_root_cause`.

Metadata: `trajectory_id`, `task_id`, `step_index`, `normalized_step_position`.

## Unavailable instrumentation and limitations

- Tool latency and retrieval confidence: no observed values in supplied data; columns remain null.
- Sender/receiver similarity and handoff information loss: no paired sent/received payloads; columns remain null.
- Premature termination: no expected completion contract; column remains null rather than using outcome or fault labels.
- Synthetic success/failure rates: generally unavailable because step statuses are absent. Observable error counts are not claims of successful execution.
- Retry counts count explicit RETRY events only, not inferred repeated actions.
- Original-task drift is unavailable where the original task-bearing step is missing. Consecutive-step similarity can still be computed.
- Long text uses MiniLM's native truncation. Final semantic aggregates refer to the final meaningful text step.
- Tool response matching follows the sequential recorder. Concurrent calls would need explicit call IDs.
- Grouping cannot discover shared provenance that the dataset never records. Synthetic task IDs are unique despite reused templates.
- All-missing training numeric columns are retained by preprocessing with sklearn's zero fallback at the encoded stage only; raw CSVs remain null.

See feature_engineering/README.md for extraction conventions and data/features/feature_dictionary.json for every column's role, level, type, description, formula, and missing condition.
