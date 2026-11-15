# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict
import pytest

from resources_servers.ifgym import generate_examples
from resources_servers.ifgym.episode import Episode
from resources_servers.ifgym.app import IFGymServer
from resources_servers.ifgym import verify_rollout as verifier
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent
from responses_api_agents.ifgym_agent.app import IFGymAgent


def action(*, status, reason=None, input_tokens=None, output_tokens=None, metadata=None):
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        usage=(
            SimpleNamespace(input_tokens=input_tokens or 0, output_tokens=output_tokens)
            if output_tokens is not None
            else None
        ),
        metadata=metadata or {},
        output=[],
    )


def test_context_exhaustion_preserves_completed_turns_without_scoring_another():
    episode = SimpleNamespace(
        sample_id="sample", rewards=[1.0, 0.5], details=[{"turn_idx": 1}, {"turn_idx": 2}],
        step=MagicMock(),
    )
    server = SimpleNamespace(session_state={"session": episode})
    result = asyncio.run(IFGymServer.step(
        server, action(status="incomplete", reason="max_output_tokens"), {}, "session"
    ))
    observation, reward, terminated, truncated, info = result
    assert observation is None and reward == 0 and not terminated and truncated
    assert info["turn_rewards"] == [1.0, 0.5]
    assert info["turn_details"] == episode.details
    assert info["truncation_reason"] == "context_exhausted"
    episode.step.assert_not_called()


def test_first_turn_context_exhaustion_has_no_turn_rewards():
    episode = SimpleNamespace(sample_id="sample", rewards=[], details=[], step=MagicMock())
    server = SimpleNamespace(session_state={"session": episode})
    _, _, _, truncated, info = asyncio.run(IFGymServer.step(
        server, action(status="incomplete", reason="max_output_tokens", output_tokens=0), {}, "session"
    ))
    assert truncated and info["turn_rewards"] == []
    episode.step.assert_not_called()


def test_generated_empty_answer_is_still_scored():
    episode = SimpleNamespace(step=MagicMock(return_value=(None, 0.0, True, {"turn_rewards": [0.0]})))
    server = SimpleNamespace(session_state={"session": episode})
    result = asyncio.run(IFGymServer.step(
        server, action(status="completed", output_tokens=1), {}, "session"
    ))
    assert result[1:4] == (0.0, True, False)
    episode.step.assert_called_once_with("")


def test_empty_response_with_generated_tokens_at_output_cap_is_scored():
    episode = SimpleNamespace(step=MagicMock(return_value=(None, 0.0, True, {"turn_rewards": [0.0]})))
    server = SimpleNamespace(session_state={"session": episode})
    asyncio.run(IFGymServer.step(
        server, action(status="incomplete", reason="max_output_tokens", output_tokens=1), {}, "session"
    ))
    episode.step.assert_called_once_with("")


def test_nonempty_response_at_output_cap_is_scored():
    episode = SimpleNamespace(step=MagicMock(return_value=(None, 1.0, True, {"turn_rewards": [1.0]})))
    server = SimpleNamespace(session_state={"session": episode})
    response = action(status="incomplete", reason="max_output_tokens", output_tokens=2048)
    response.output = [SimpleNamespace(type="message", content="answer")]
    asyncio.run(IFGymServer.step(server, response, {}, "session"))
    episode.step.assert_called_once_with("answer")


@pytest.mark.parametrize("total_tokens", [32_768, 32_769])
def test_nonempty_response_reaching_sequence_limit_is_dropped(total_tokens):
    episode = SimpleNamespace(
        sample_id="sample", rewards=[1.0], details=[{"turn_idx": 1}], step=MagicMock()
    )
    server = SimpleNamespace(session_state={"session": episode})
    response = action(
        status="incomplete",
        reason="max_output_tokens",
        input_tokens=total_tokens - 2048,
        output_tokens=2048,
    )
    response.output = [SimpleNamespace(type="message", content="discard this turn")]

    observation, reward, terminated, truncated, info = asyncio.run(
        IFGymServer.step(server, response, {}, "session")
    )

    assert observation is None and reward == 0 and not terminated and truncated
    assert info["turn_rewards"] == [1.0]
    assert info["turn_details"] == episode.details
    assert info["truncation_reason"] == "context_exhausted"
    episode.step.assert_not_called()


def test_nonempty_2048_token_completion_below_sequence_limit_is_scored():
    episode = SimpleNamespace(step=MagicMock(return_value=(None, 1.0, True, {"turn_rewards": [1.0]})))
    server = SimpleNamespace(session_state={"session": episode})
    response = action(
        status="incomplete", reason="max_output_tokens", input_tokens=30_719, output_tokens=2048
    )
    response.output = [SimpleNamespace(type="message", content="valid capped answer")]

    result = asyncio.run(IFGymServer.step(server, response, {}, "session"))

    assert result[1:4] == (1.0, True, False)
    episode.step.assert_called_once_with("valid capped answer")


def test_context_window_metadata_marks_prompt_overflow_before_output():
    episode = SimpleNamespace(sample_id="sample", rewards=[], details=[], step=MagicMock())
    server = SimpleNamespace(session_state={"session": episode})
    response = action(status="incomplete", reason="max_output_tokens", metadata={
        "nemo_gym_context_window_exceeded": "true",
    })

    _, _, _, truncated, info = asyncio.run(IFGymServer.step(server, response, {}, "session"))

    assert truncated and info["truncation_reason"] == "context_exhausted"
    episode.step.assert_not_called()


def test_first_turn_context_exhaustion_is_flagged_for_masking(monkeypatch):
    result = SimpleNamespace(info={"turn_rewards": [], "truncation_reason": "context_exhausted"})
    result.model_copy = MagicMock(return_value=result)

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))
    assert result.model_copy.call_args.kwargs["update"] == {
        "reward": 0.0, "instance_config": {"mask_sample": True}, "context_exhausted": True
    }


def test_first_turn_exhaustion_removes_all_response_output(monkeypatch):
    class Response(BaseModel):
        model_config = ConfigDict(extra="allow")
        output: list[dict]

    class Rollout(BaseModel):
        model_config = ConfigDict(extra="allow")
        info: dict
        response: Response
        reward: float = 0.0
        instance_config: dict = {}

    result = Rollout(
        info={"turn_rewards": [], "truncation_reason": "context_exhausted"},
        response=Response(output=[{"role": "assistant", "content": ""}]),
    )

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    actual = asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))

    assert actual.response.output == []
    assert actual.instance_config == {"mask_sample": True}


def test_partial_rollout_keeps_mean_of_completed_rewards(monkeypatch):
    result = SimpleNamespace(info={"turn_rewards": [1.0, 0.5], "truncation_reason": "context_exhausted"})
    result.model_copy = MagicMock(return_value=result)

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))
    assert result.model_copy.call_args.kwargs["update"] == {"reward": 0.75, "context_exhausted": True}


def test_partial_rollout_prunes_over_budget_prompt_and_entire_final_turn(monkeypatch):
    class Response(BaseModel):
        model_config = ConfigDict(extra="allow")
        output: list[dict]

    class Rollout(BaseModel):
        model_config = ConfigDict(extra="allow")
        info: dict
        response: Response
        reward: float = 0.0

    result = Rollout(
        info={"turn_rewards": [1.0, 0.5], "truncation_reason": "context_exhausted"},
        response=Response(output=[
            {"type": "message", "role": "assistant", "content": "turn one"},
            {"type": "message", "role": "user", "content": "turn two prompt"},
            {"type": "message", "role": "assistant", "content": "turn two"},
            {"type": "message", "role": "user", "content": "over-budget prompt"},
            {"type": "reasoning", "content": "discarded reasoning"},
            {"type": "message", "role": "assistant", "content": "discarded completion"},
        ]),
    )

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    actual = asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))

    assert actual.response.output == result.response.output[:3]
    assert actual.reward == 0.75


def test_unrecognizable_prefix_is_masked_instead_of_returning_empty_unmasked_rollout(monkeypatch):
    class Response(BaseModel):
        model_config = ConfigDict(extra="allow")
        output: list[dict]

    class Rollout(BaseModel):
        model_config = ConfigDict(extra="allow")
        info: dict
        response: Response
        reward: float = 1.0
        instance_config: dict = {}

    result = Rollout(
        info={"turn_rewards": [1.0], "turn_details": [{"turn_idx": 1}], "truncation_reason": "context_exhausted"},
        response=Response(output=[{"type": "message", "role": "assistant", "content": "unknown boundary"}]),
    )

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    actual = asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))

    assert actual.response.output == []
    assert actual.info["turn_rewards"] == []
    assert actual.info["turn_details"] == []
    assert actual.reward == 0.0
    assert actual.instance_config["mask_sample"] is True
    assert actual.mask_sample is True


def test_pruned_context_exhaustion_transcript_verifies_against_completed_prefix(monkeypatch):
    class Response(BaseModel):
        model_config = ConfigDict(extra="allow")
        output: list[dict]

    class Rollout(BaseModel):
        model_config = ConfigDict(extra="allow")
        info: dict
        response: Response
        responses_create_params: dict
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = True

    row = generate_examples.examples()[0]
    episode = Episode(row["episode"])
    first_prompt = episode.turns[0]["rendered_prompt"]
    second_prompt, _, _, _ = episode.step("answer one")
    _, _, _, info = episode.step("answer two")
    third_prompt = episode.turns[2]["rendered_prompt"]
    result = Rollout(
        info=info | {"truncation_reason": "context_exhausted"},
        responses_create_params=row["responses_create_params"],
        response=Response(output=[
            {"type": "message", "role": "assistant", "content": "answer one"},
            {"type": "message", "role": "user", "content": second_prompt},
            {"type": "message", "role": "assistant", "content": "answer two"},
            {"type": "message", "role": "user", "content": third_prompt},
            {"type": "reasoning", "content": "discarded"},
            {"type": "message", "role": "assistant", "content": "discarded"},
        ]),
    )

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    actual = asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))

    assert first_prompt == row["episode"]["turns"][0]["rendered_prompt"]
    assert actual.response.output[-1]["content"] == "answer two"
    verifier.verify_rollout(row, actual.model_dump())


def test_completed_rollout_reports_no_context_exhaustion(monkeypatch):
    class Response(BaseModel):
        model_config = ConfigDict(extra="allow")
        info: dict
        reward: float = 0.0

    result = Response(info={"turn_rewards": [1.0]})

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    actual = asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))
    assert actual.model_dump()["context_exhausted"] is False
    assert actual.reward == 1.0


def test_exhausted_rollout_serializes_metric_for_aggregation(monkeypatch):
    class Response(BaseModel):
        model_config = ConfigDict(extra="allow")
        info: dict
        reward: float = 0.0

    result = Response(info={"turn_rewards": [1.0], "truncation_reason": "context_exhausted"})

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    actual = asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))
    assert actual.model_dump()["context_exhausted"] is True


def test_verifier_accepts_masked_first_turn_exhaustion(monkeypatch):
    episode = SimpleNamespace(turns=[{"rendered_prompt": "prompt"}], rewards=[], index=0)
    monkeypatch.setattr(verifier, "Episode", lambda _: episode)
    monkeypatch.setattr(verifier, "initial_user_text", lambda _: "prompt")
    row = {"episode": {"sample_id": "sample"}, "responses_create_params": {}}
    result = {
        "responses_create_params": {}, "response": {"output": []}, "reward": 0.0,
        "terminated": False, "truncated": True, "instance_config": {"mask_sample": True},
        "info": {"sample_id": "sample", "turn_details": [], "turn_rewards": [],
                 "truncation_reason": "context_exhausted"},
    }
    verifier.verify_rollout(row, result)
