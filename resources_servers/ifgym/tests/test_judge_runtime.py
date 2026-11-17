# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


if sys.version_info < (3, 13):
    pytest.skip("NeMo Gym requires Python 3.13", allow_module_level=True)

import httpx
from ifgym.instructions.instructions_registry import INSTRUCTION_DICT, get_instruction_metadata


try:
    from nemo_gym.client import SESSION_ID_KEY, ServerClient
    from nemo_gym.config import ModelServerRef
    from nemo_gym.core import NeMoGymResponse
except ImportError:
    from nemo_gym.config_types import ModelServerRef
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient

from resources_servers.gymnasium import EnvResetRequest, EnvStepRequest
from resources_servers.ifgym.app import IFGymResourcesServerConfig, IFGymServer
from resources_servers.ifgym.judge import JudgeResult, UnresolvedGradeError


DATA = Path(__file__).parents[1] / "data/example.jsonl"
SEMANTIC_ID = "nonverifiable:professional_tone"


def semantic_row(*, one_turn=False):
    row = json.loads(DATA.read_text().splitlines()[0])
    episode = row["episode"]
    if one_turn:
        episode["turns"] = episode["turns"][:1]
        episode["n_turns"] = 1
    handle = episode["turns"][0]["active_constraints"][0]["handle"]
    for turn in episode["turns"]:
        for item in turn["events"] + turn["active_constraints"]:
            if item.get("handle") == handle:
                item["constraint_id"] = SEMANTIC_ID
                item["kwargs"] = {}
    meta = get_instruction_metadata(SEMANTIC_ID)
    shown = INSTRUCTION_DICT[SEMANTIC_ID](SEMANTIC_ID).build_description()
    episode["metadata"]["non_verifiable"] = {
        "schema_version": 1,
        "registry_revision": "test",
        "annotations_sha256": hashlib.sha256(b"annotations").hexdigest(),
        "paraphrases_sha256": hashlib.sha256(b"paraphrases").hexdigest(),
        "constraints_by_handle": {
            handle: {
                "constraint_id": SEMANTIC_ID,
                "verifier_type": "llm_judge",
                "rubric_version": meta.rubric_version,
                "rubric_sha256": meta.rubric_sha256,
                "instruction_as_shown": shown,
                "introduced_at": episode["turns"][0]["turn_idx"],
            }
        },
    }
    row["responses_create_params"]["input"] = [{"role": "user", "content": episode["turns"][0]["rendered_prompt"]}]
    return row, handle


def model_response(text, *, metadata=None):
    return NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0.0,
            "model": "mock",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "metadata": metadata or {},
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
        }
    )


class GateJudge:
    def __init__(self, *, gate=False, error=None):
        self.calls = 0
        self.gate = gate
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.error = error

    async def score(self, _context):
        self.calls += 1
        self.entered.set()
        if self.gate:
            await self.release.wait()
        if self.error:
            raise self.error
        return JudgeResult("pass", 1, {"request_sha256": "unit-digest"})


def make_server(*, gate=False, error=None):
    config = IFGymResourcesServerConfig(
        host="",
        port=0,
        entrypoint="app.py",
        name="ifgym_resources_server",
        max_total_sequence_length=32768,
        judge_enabled=True,
        judge_model_server=ModelServerRef(type="responses_api_models", name="ifgym_judge_model"),
    )
    server = IFGymServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._judge_adapter = GateJudge(gate=gate, error=error)
    return server, server._judge_adapter


def test_pending_duplicate_shares_grade_and_receipt_is_deep_copied():
    async def run():
        row, _handle = semantic_row()
        server, judge = make_server(gate=True)
        await server.reset({"episode": row["episode"]}, "session")
        action = model_response("A respectful answer.")
        metadata = {"_ng_step_request_id": "same-step"}
        first = asyncio.create_task(server.step(action, metadata, "session"))
        await judge.entered.wait()
        duplicate = asyncio.create_task(server.step(action, metadata, "session"))
        await asyncio.sleep(0)
        judge.release.set()
        first_result, duplicate_result = await asyncio.gather(first, duplicate)
        assert judge.calls == 1
        assert first_result == duplicate_result
        first_result[4]["turn_rewards"].append(999)
        replay = await server.step(action, metadata, "session")
        assert replay[4]["turn_rewards"] == [1.0]
        assert server.session_state["session"].index == 1

    asyncio.run(run())


def test_unresolved_grade_does_not_commit_any_episode_state():
    async def run():
        row, _handle = semantic_row()
        server, _judge = make_server(error=UnresolvedGradeError("uncertain", "no grade"))
        await server.reset({"episode": row["episode"]}, "session")
        episode = server.session_state["session"]
        with pytest.raises(UnresolvedGradeError):
            await server.step(
                model_response("A respectful answer."),
                {"_ng_step_request_id": "unresolved"},
                "session",
            )
        assert episode.index == 0
        assert episode.responses == []
        assert episode.rewards == []
        assert episode.details == []

    asyncio.run(run())


def test_unresolved_step_receipt_prevents_same_id_rejudging():
    async def run():
        row, _handle = semantic_row()
        server, judge = make_server(error=UnresolvedGradeError("uncertain", "no grade"))
        await server.reset({"episode": row["episode"]}, "session")
        episode = server.session_state["session"]
        action = model_response("A respectful answer.")
        metadata = {"_ng_step_request_id": "unresolved-retry"}
        for _ in range(2):
            with pytest.raises(UnresolvedGradeError, match="no grade"):
                await server.step(action, metadata, "session")
        assert judge.calls == 1
        assert episode.index == 0
        assert episode.responses == []
        assert episode.rewards == []
        assert episode.details == []


@pytest.mark.parametrize("shutdown", ["reset", "close"])
def test_reset_or_close_cancels_pending_judge_before_commit(shutdown):
    async def run():
        row, _handle = semantic_row()
        server, judge = make_server(gate=True)
        await server.reset({"episode": row["episode"]}, "session")
        old = server.session_state["session"]
        pending = asyncio.create_task(
            server.step(model_response("A respectful answer."), {"_ng_step_request_id": "pending"}, "session")
        )
        await judge.entered.wait()
        if shutdown == "reset":
            await server.reset({"episode": row["episode"]}, "session")
            new = server.session_state["session"]
            assert new is not old and new.index == 0
        else:
            await server.close_session("session")
            assert "session" not in server.session_state
        result = await asyncio.gather(pending, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert old.index == 0
        assert old.responses == []

    asyncio.run(run())


def test_terminal_receipt_replays_after_close_and_old_cookie_cannot_close_new_episode():
    async def run():
        row, _handle = semantic_row(one_turn=True)
        server, judge = make_server()
        old_session = "old-cookie"
        request = SimpleNamespace(session={SESSION_ID_KEY: old_session})
        reset_body = EnvResetRequest.model_validate(row)
        await server._reset_endpoint(reset_body, request)
        episode_session = request.session[SESSION_ID_KEY]
        action = model_response("A respectful answer.")
        step_body = EnvStepRequest.model_validate(
            {**row, "response": action.model_dump(mode="json"), "_ng_step_request_id": "terminal-step"}
        )
        first = await server._step_endpoint(step_body, request)
        assert first.terminated is True
        assert episode_session not in server.session_state
        replay = await server._step_endpoint(step_body, request)
        assert replay == first
        assert judge.calls == 1

        await server._reset_endpoint(reset_body, request)
        new_session = request.session[SESSION_ID_KEY]
        assert new_session != episode_session
        assert new_session in server.session_state
        old_cookie = SimpleNamespace(session={SESSION_ID_KEY: episode_session})
        replay_after_reset = await server._step_endpoint(step_body, old_cookie)
        assert replay_after_reset == first
        assert new_session in server.session_state
        assert server.session_state[new_session].index == 0

    asyncio.run(run())


def test_truncated_receipt_replays_after_base_closes_session():
    async def run():
        row, _handle = semantic_row(one_turn=True)
        server, judge = make_server()
        request = SimpleNamespace(session={SESSION_ID_KEY: "truncated-cookie"})
        await server._reset_endpoint(EnvResetRequest.model_validate(row), request)
        sid = request.session[SESSION_ID_KEY]
        action = model_response("unfinished", metadata={"nemo_gym_context_window_exceeded": "true"})
        body = EnvStepRequest.model_validate(
            {**row, "response": action.model_dump(mode="json"), "_ng_step_request_id": "truncated-step"}
        )
        first = await server._step_endpoint(body, request)
        assert first.truncated is True
        assert first.reward == 0.0
        assert sid not in server.session_state
        replay = await server._step_endpoint(body, request)
        assert replay == first
        assert judge.calls == 0

    asyncio.run(run())


def test_http_cookie_rotation_terminal_retry_and_old_cookie_replay():
    async def run():
        row, _handle = semantic_row(one_turn=True)
        server, judge = make_server()
        transport = httpx.ASGITransport(app=server.setup_webserver())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            reset = await client.post("/reset", json=row)
            assert reset.status_code == 200
            old_session = next(iter(server.session_state))
            old_cookies = httpx.Cookies(client.cookies)
            step_payload = {
                **row,
                "response": model_response("A respectful answer.").model_dump(mode="json"),
                "_ng_step_request_id": "terminal-http-step",
            }
            first = await client.post("/step", json=step_payload)
            assert first.status_code == 200
            assert first.json()["terminated"] is True
            assert old_session not in server.session_state
            replay = await client.post("/step", json=step_payload)
            assert replay.json() == first.json()
            assert judge.calls == 1

            second_reset = await client.post("/reset", json=row)
            assert second_reset.status_code == 200
            new_session = next(iter(server.session_state))
            assert new_session != old_session
            old_cookie_replay = await client.post("/step", json=step_payload, cookies=old_cookies)
            assert old_cookie_replay.json() == first.json()
            assert new_session in server.session_state
            assert server.session_state[new_session].index == 0
            assert judge.calls == 1

    asyncio.run(run())


def test_http_truncated_response_is_cached_after_base_cleanup():
    async def run():
        row, _handle = semantic_row(one_turn=True)
        server, judge = make_server()
        transport = httpx.ASGITransport(app=server.setup_webserver())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/reset", json=row)
            sid = next(iter(server.session_state))
            action = model_response("unfinished", metadata={"nemo_gym_context_window_exceeded": "true"})
            payload = {
                **row,
                "response": action.model_dump(mode="json"),
                "_ng_step_request_id": "truncated-http-step",
            }
            first = await client.post("/step", json=payload)
            assert first.status_code == 200
            assert first.json()["truncated"] is True
            assert sid not in server.session_state
            replay = await client.post("/step", json=payload)
            assert replay.json() == first.json()
            assert judge.calls == 0

    asyncio.run(run())


def test_request_id_rejects_changed_payload_and_new_episode_reuse():
    async def run():
        row, _handle = semantic_row()
        server, _judge = make_server()
        await server.reset({"episode": row["episode"]}, "session")
        request_id = {"_ng_step_request_id": "bound-step"}
        await server.step(model_response("A respectful answer."), request_id, "session")
        with pytest.raises(ValueError, match="different response payload"):
            await server.step(model_response("A different answer."), request_id, "session")
        assert server.session_state["session"].index == 1

        await server.reset({"episode": row["episode"]}, "session")
        with pytest.raises(ValueError, match="different IFGym episode"):
            await server.step(model_response("A respectful answer."), request_id, "session")
        assert server.session_state["session"].index == 0

    asyncio.run(run())


def test_deterministic_episode_does_not_call_judge_when_semantic_mode_enabled():
    async def run():
        row = json.loads(DATA.read_text().splitlines()[0])
        server, judge = make_server()
        await server.reset({"episode": row["episode"]}, "deterministic-session")
        await server.step(
            model_response("This response follows the requested instruction."),
            {"_ng_step_request_id": "deterministic-step"},
            "deterministic-session",
        )
        assert judge.calls == 0

    asyncio.run(run())
