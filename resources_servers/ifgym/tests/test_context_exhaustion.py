# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict

from resources_servers.ifgym.app import IFGymServer
from resources_servers.ifgym import verify_rollout as verifier
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent
from responses_api_agents.ifgym_agent.app import IFGymAgent


def action(*, status, reason=None, output_tokens=None):
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        usage=SimpleNamespace(output_tokens=output_tokens) if output_tokens is not None else None,
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


def test_partial_rollout_keeps_mean_of_completed_rewards(monkeypatch):
    result = SimpleNamespace(info={"turn_rewards": [1.0, 0.5], "truncation_reason": "context_exhausted"})
    result.model_copy = MagicMock(return_value=result)

    async def run_base(*args):
        return result

    monkeypatch.setattr(GymnasiumAgent, "run", run_base)
    asyncio.run(IFGymAgent.run(IFGymAgent.__new__(IFGymAgent), None, None))
    assert result.model_copy.call_args.kwargs["update"] == {"reward": 0.75, "context_exhausted": True}


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
