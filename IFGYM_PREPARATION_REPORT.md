# IFGym preparation report

## Repository state

- NeMo Gym branch: `ifgym`
- Pre-cleanup commit: `b4dc559825a951171c0218623f1c44243ceb7843`
- Scope: this NeMo Gym checkout only. The sibling IFGym source and `/users/emma_anastassova/if-gym/train/` were inspected read-only and were not changed.
- Shared `nemo_gym/` runtime and `responses_api_models/` remain intact.

## Existing IFGym behavior

IFGym registers 78 deterministic instruction checks, 24 of which inspect conversation history. The checks cover response format and JSON, keyword and phrase requirements, length and counts, language, casing and punctuation, copying and repetition, first/last words, and history conditions. The verifier path does not use an LLM judge. LLM calls are used in data generation for prompt chains and paraphrases.

Generated conversations store turn prompts, constraint introduction/revocation schedules, active rules, rendered layouts, split, and metadata. Constraints can begin now or later, recur, persist until revoked, and change over a conversation. The generator caps active constraints at five and retries invalid schedules/conflicts. Same-turn incompatible constraints are rejected; there is no priority hierarchy that resolves them.

Single-turn reward is the mean of check results, so multiple boolean checks yield fractional rewards. Unknown or unbuildable checks are skipped; blank answers fail; verifier exceptions count as failures; no checked constraints yields zero. Multi-turn scoring averages active-constraint scores per turn and then averages turn scores for the trajectory. The VeRL multi-turn integration also has custom per-turn advantage and RLOO code. These schemas, schedules, and checkers are trainer-independent; the current rollout and reward plumbing is VeRL-specific. No files in that implementation were edited.

## NeMo Gym architecture recommendation

For an initial single-turn version, port the deterministic checks behind a custom Resources Server, use JSONL rows carrying the rendered prompt and constraint metadata, and use the standard single-turn environment path. Keep IFGym's mean-of-checks reward and expose individual check results as diagnostics/reward components. Do not assume NeMo Gym's stock instruction-following reward has identical empty-list, blank-output, exception, or denominator behavior.

For genuine multi-turn episodes, use a custom Environment Server to own conversation state, turn progression, active constraints, introduction/revocation events, and turn rewards. `example_multi_turn_gymnasium` and `nemo_user_sim` are the strongest lifecycle references; `example_session_state_mgmt` shows persisted session state. The conversational and agentic simulation servers are useful interaction references, but their scalar terminal rewards do not provide IFGym's turn-local credit assignment.

NeMo-RL's current NeMo Gym adapter reads the final scalar from `reward`; `reward_components` become per-sample objective columns and must sum to that scalar. This does not map a turn reward onto the tokens produced on that turn. IFGym's multi-turn GRPO/RLOO therefore needs a NeMo-RL adapter/trainer change that preserves each assistant token span and maps each turn score to that span. The adapter also expects token-prefix continuity across turns. These details were checked against upstream `main`, which is not pinned to a local NeMo-RL revision; confirm them at the revision selected for integration: [NeMo Gym adapter](https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main/nemo_rl/environments/nemo_gym.py), [rollout processing](https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main/nemo_rl/experience/rollouts.py).

## Retained components

The primary instruction-following and verifier set is `instruction_following`, `ifbench`, `multichallenge`, `format_verification`, `structured_outputs`, `inverse_if`, `agentif`, `iheval`, `agentic_if`, `partner_if`, `equivalence_rule`, and `equivalence_llm_judge`. Also retained are the requested multi-turn and reward references: `example_multi_turn_gymnasium`, `example_session_state_mgmt`, `nemo_user_sim`, `conversational_tool_use_simulation`, `agentic_turn_simulation`, `example_tool_call_multireward`, and `example_multi_step`. `citation_if` and `verifif` remain as adjacent verifier references.

Environment Servers retained: `harbor`, `legacy_agent`, `nemo_user_sim`, `single_agent_turn`, and `single_agent_turn_legacy`. Top-level environment packages retained are `instruction_following`, `nemo_user_sim`, `harbor`, and the LangGraph reflection/parallel/orchestrator/ReWOO math and reasoning examples.

The cleanup retains dependency-only components when active configurations, runtime imports, or shared tests require them. Examples include Harbor agents and dataset fixtures, `math_with_judge`/`reasoning_gym`, OSWorld lock/test fixtures, NoOA's SWE-bench test dependencies, benchmark configuration fixtures used by shared CLI/composition tests, and the single-step tool-use server referenced by repository tooling. These are test or dependency support, not proposed IFGym task targets. The full retained inventory is visible under the five component roots in this checkout.

## Cleanup summary

Top-level package directories removed from the pre-cleanup tree:

- 134 benchmark directories, including unrelated math, coding/SWE, science, finance, multimodal, and domain-specific suites.
- 130 Resources Server directories, including unrelated games, coding/SWE services, science, finance, multimodal, and search services.
- 47 standalone environment packages, including unrelated games, coding-agent variants, math tasks, and domain tasks.
- 33 Responses API Agent packages and one Environment Server (`swe_together`).

The requested `environments/` tree was audited for fixture and config references before pruning. Retained shared infrastructure and every retained YAML `config_paths` dependency resolve in the static audit. A single stale OSWorld profile that named two config files absent from the pre-cleanup commit was removed. Benchmark-specific tests and recipe files were pruned alongside their unrelated suites; shared discovery, routing, and validation checks now use retained IF fixtures. The Nemotron 3.5 Super suite configs were removed; its Slurm launcher remains as a fixture for the shared OpenSandbox cleanup test.

## Validation

- Parsed 400 retained YAML files: no parse errors and no unresolved `config_paths`.
- Scanned Python component imports: no missing local package paths. The AST scan found eight files that Python 3.11 cannot parse because the repository uses newer Python syntax; the project requires Python 3.13.14 or newer.
- Compiled all six edited Python test files with Python 3.11 successfully.
- `git diff --check HEAD` passed.
- The focused pytest run used pytest 9.0.3 from the `ifgym` Conda environment and stopped while loading `tests/unit_tests/conftest.py`: `omegaconf` is not installed. That environment uses Python 3.11.15, below the repository requirement of Python 3.13.14 or newer. Server import/launch checks also need the project’s declared dependencies. No dependencies were installed.

The cleanup is statically consistent, but the full runtime test suite and live server discovery still need to be run in the repository's supported Python/dependency environment.
