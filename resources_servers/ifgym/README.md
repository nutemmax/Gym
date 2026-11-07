# IFGym resource server

The IFGym resource server runs scripted, multi-turn instruction-following episodes in NeMo Gym. It uses the `ifgym` package for conversation schedules and constraint checkers. The companion `ifgym_agent` sends each prompt to the model and returns the mean turn reward for the episode.

## Dataset

Each JSONL row contains `id`, `agent_ref`, `responses_create_params.input`, and `episode`. The initial user message must match `episode.turns[0].rendered_prompt`. The episode uses IFGym's `ConversationInternalRepr` serialization and includes a sample ID, turn prompts, constraint events, and active-constraint snapshots.

At reset, the server rebuilds the constraint schedule and checks the snapshots, constraint IDs, and required checker arguments. At each step it scores the assistant response, records the result, and returns the next prompt. It releases the session when the episode completes or the client closes it.

`data/example.jsonl` contains synthetic examples. To convert IFGym v3 data to the Gym format, run `prepare_nemogym_dataset.py --help` from the IFGym workspace. The examples are not a training split.

## Rewards and truncation

The reward for each turn is the fraction of active checks that pass. A checker that cannot be built is skipped; a checker that raises while evaluating an answer counts as failed. If no check runs, the reward is zero. An empty generated answer fails the applicable checks.

`info.turn_rewards` contains the per-turn rewards. `info.turn_details` contains prompts, responses, active constraints, checker results, and errors. The agent reports the mean of the completed turn rewards as `reward`.

When the model server reports context exhaustion without generating tokens, the server truncates the episode before scoring the nonexistent answer. It preserves completed turns and sets `info.truncation_reason` to `context_exhausted`. If no turn was generated, the agent flags the rollout for loss masking. The supplied agent configuration allows up to 10 turns; reaching that limit can also truncate an episode.

Session state is in memory, and `/step` does not deduplicate retried requests. A retry can therefore change episode state more than once. Account for this limitation when using request retries.

## Training integration

The server requires the sibling IFGym repository and its dependencies. The local `pyproject.toml` points to that workspace. The `rl-infra` Apertus 2 backend stages the server, agent, and checker package into its pinned Gym checkout before launch. Do not stage files into a checkout used by a running job.

NeMo-RL's stock Gym adapter uses a trajectory-level reward. IFGym per-turn GRPO additionally requires the `rl-infra` reward-transport and advantage-estimator patches. See `rl-infra/IFGYM_TRAINING.md` and `rl-infra/IFGYM_PIPELINE.md` in the IFGym workspace for configuration and launch instructions.

When rollout logging is enabled, the full JSONL results contain readable prompts, responses, and checker details. Treat these files as containing dataset and model output. To replay recorded checks against the source dataset, run from the Gym root:

```bash
PYTHONPATH=.. python -m resources_servers.ifgym.verify_rollout \
  --dataset /path/to/dataset.jsonl \
  --rollouts /path/to/full-rollouts.jsonl
```

For IFGym training, `train/ifgym_agent/context_exhausted/mean` is the fraction of rollouts truncated for context exhaustion. The generic NeMo Gym truncation metric uses a different calculation and can miss this condition. Integrated validation of this metric and the truncation path remains necessary.
