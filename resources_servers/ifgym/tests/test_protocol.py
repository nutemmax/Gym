# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


if sys.version_info < (3, 13):
    pytest.skip("NeMo Gym requires Python 3.13", allow_module_level=True)

import httpx
from omegaconf import OmegaConf


try:
    from nemo_gym.client import SESSION_ID_KEY, ServerClient
    from nemo_gym.config import ModelServerRef, ResourcesServerRef
    from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from nemo_gym.core import NeMoGymResponse
    from nemo_gym.environment.resources_server_registry import discover_resources_servers
    from nemo_gym.server_runtime import BaseResourcesServerConfig

    HAS_ENVIRONMENT_SERVERS = True
except ModuleNotFoundError:
    from nemo_gym.base_resources_server import BaseResourcesServerConfig
    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient

    HAS_ENVIRONMENT_SERVERS = False
from resources_servers.gymnasium import EnvResetRequest, EnvStepRequest
from resources_servers.ifgym.app import IFGymServer
from resources_servers.ifgym.verify_rollout import verify_rollout
from responses_api_agents.gymnasium_agent.app import GymnasiumAgentConfig, GymnasiumAgentRunRequest
from responses_api_agents.ifgym_agent.app import IFGymAgent


def model_response(text):
    return NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0.0,
            "model": "mock",
            "object": "response",
            "output": [
                {
                    "id": "msg",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
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


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    async def json(self):
        return self.payload

    async def read(self):
        return json.dumps(self.payload).encode()

    def raise_for_status(self):
        return None


@pytest.mark.skipif(not HAS_ENVIRONMENT_SERVERS, reason="environment server registry is not in the pinned Gym")
def test_server_discovery_and_config_composition():
    assert discover_resources_servers()["ifgym"].config_path.name == "ifgym.yaml"
    config = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [
                "resources_servers/ifgym/configs/ifgym.yaml",
                "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            ],
            "policy_base_url": "http://127.0.0.1:1/v1",
            "policy_api_key": "dummy",
            "policy_model_name": "mock",
            "error_on_agent_without_environment_server": True,
        },
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=config,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert resolved.ifgym_agent.responses_api_agents.ifgym_agent.resources_server.name == "ifgym_resources_server"
    assert resolved.ifgym_environment_server.environment_servers.legacy_agent.agent_server.name == "ifgym_agent"


def test_reset_rejects_mismatched_initial_prompt():
    async def run():
        row = json.loads((Path(__file__).parents[1] / "data/example.jsonl").read_text().splitlines()[0])
        row["responses_create_params"]["input"][0]["content"] = "different prompt"
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="ifgym_resources_server")
        server = IFGymServer(config=config, server_client=MagicMock(spec=ServerClient))
        with pytest.raises(ValueError, match="initial user prompt"):
            await server._reset_endpoint(
                EnvResetRequest.model_validate(row), SimpleNamespace(session={SESSION_ID_KEY: "session"})
            )
        assert not server.session_state

    asyncio.run(run())


def test_sessions_keep_independent_constraint_state():
    async def run():
        row = json.loads((Path(__file__).parents[1] / "data/example.jsonl").read_text().splitlines()[0])
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="ifgym_resources_server")
        server = IFGymServer(config=config, server_client=MagicMock(spec=ServerClient))
        await server.reset({"episode": row["episode"]}, "first")
        await server.reset({"episode": row["episode"]}, "second")
        _, first_reward, _, _, first_info = await server.step(model_response("seed helps plants"), {}, "first")
        _, second_reward, _, _, second_info = await server.step(model_response("seed helps plants"), {}, "second")
        assert first_reward == second_reward == 1.0
        assert first_info["turn_details"][0]["turn_idx"] == second_info["turn_details"][0]["turn_idx"] == 1
        await server.close_session("first")
        assert "first" not in server.session_state and "second" in server.session_state
        await server.close_session("second")
        assert not server.session_state

    asyncio.run(run())


def test_http_session_cookies_and_explicit_close():
    async def run():
        row = json.loads((Path(__file__).parents[1] / "data/example.jsonl").read_text().splitlines()[0])
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="ifgym_resources_server")
        server = IFGymServer(config=config, server_client=MagicMock(spec=ServerClient))
        transport = httpx.ASGITransport(app=server.setup_webserver())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            reset = await client.post("/reset", json=row)
            assert reset.status_code == 200
            assert reset.json()["info"]["supports_explicit_close"]
            assert len(server.session_state) == 1
            step = await client.post(
                "/step", json={**row, "response": model_response("seed helps plants").model_dump(mode="json")}
            )
            assert step.status_code == 200
            assert step.json()["reward"] == 1.0
            assert (await client.post("/close", json={})).status_code == 200
            assert not server.session_state
            assert (await client.post("/close", json={})).status_code == 200

    asyncio.run(run())


@pytest.mark.parametrize("max_steps", [2, 5])
def test_mock_model_rollout_and_session_cleanup(max_steps):
    async def run():
        row = json.loads((Path(__file__).parents[1] / "data/example.jsonl").read_text().splitlines()[0])
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="ifgym_resources_server")
        server = IFGymServer(config=config, server_client=MagicMock(spec=ServerClient))
        request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {"observability_enabled": False}
        responses = ["seed helps plants", "seed, bloom", "flower helps plants", "flower All done."]
        model_outputs = iter(responses)

        async def post(*, server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path == "/v1/responses":
                return FakeHTTPResponse(model_response(next(model_outputs)).model_dump(mode="json"))
            if url_path == "/reset":
                result = await server._reset_endpoint(EnvResetRequest.model_validate(json), request)
                return FakeHTTPResponse(result.model_dump())
            if url_path == "/step":
                result = await server._step_endpoint(EnvStepRequest.model_validate(json), request)
                return FakeHTTPResponse(result.model_dump())
            if url_path == "/close":
                return FakeHTTPResponse(await server.close_episode(request))
            raise AssertionError(url_path)

        client.post = AsyncMock(side_effect=post)
        agent_config = GymnasiumAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="ifgym_agent",
            max_steps=max_steps,
            resources_server=ResourcesServerRef(type="resources_servers", name="ifgym_resources_server"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        agent = IFGymAgent(config=agent_config, server_client=client)
        result = await agent.run(SimpleNamespace(cookies={}), GymnasiumAgentRunRequest.model_validate(row))
        expected_rewards = [1, 2 / 3, 1, 1][:max_steps]
        assert result.terminated is (max_steps >= len(responses))
        assert result.truncated is (max_steps < len(responses))
        assert result.info["turn_rewards"] == pytest.approx(expected_rewards)
        assert result.reward == pytest.approx(sum(expected_rewards) / len(expected_rewards))
        assert [item["response"] for item in result.info["turn_details"]] == responses[:max_steps]
        verify_rollout(row, result.model_dump(mode="json"))
        assert "session" not in server.session_state

        await server.reset({"episode": row["episode"]}, "cancelled")
        await server.close_episode(SimpleNamespace(session={SESSION_ID_KEY: "cancelled"}))
        assert "cancelled" not in server.session_state

    asyncio.run(run())
