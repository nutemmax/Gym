# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import FastAPI, Request


try:
    from nemo_gym.client import SESSION_ID_KEY
    from nemo_gym.core import NeMoGymResponse
except ImportError:
    # The Gym revision pinned by rl-infra predates the package reorganization.
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.gymnasium import EnvResetRequest, EnvResetResponse, GymnasiumServer, extract_text
from resources_servers.ifgym.episode import Episode, initial_user_text


class IFGymServer(GymnasiumServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close")(self.close_episode)
        return app

    async def close_episode(self, request: Request) -> dict[str, bool]:
        session_id = request.session.get(SESSION_ID_KEY)
        was_open = session_id in self.session_state
        await self.close_session(session_id)
        if was_open and hasattr(self, "checkpoint_session_ended"):
            self.checkpoint_session_ended(request)
        return {"closed": True}

    async def _reset_endpoint(self, body: EnvResetRequest, request: Request) -> EnvResetResponse:
        episode = (body.model_extra or {}).get("episode")
        if (
            not isinstance(episode, dict)
            or not isinstance(episode.get("turns"), list)
            or not episode["turns"]
            or not isinstance(episode["turns"][0], dict)
        ):
            raise ValueError("episode must contain at least one turn")
        initial_prompt = episode["turns"][0].get("rendered_prompt")
        if initial_user_text(body.responses_create_params.model_dump(mode="json")) != initial_prompt:
            raise ValueError("initial user prompt does not match the episode")
        return await super()._reset_endpoint(body, request)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[None, dict]:
        if session_id is None:
            raise ValueError("IFGym requires a session ID")
        episode = metadata.get("episode")
        if not isinstance(episode, dict):
            raise ValueError("episode metadata is required")
        await self.close_session(session_id)
        self.session_state[session_id] = Episode(episode)
        return None, {"supports_explicit_close": True}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        if session_id not in self.session_state:
            raise ValueError("IFGym session is missing; call reset first")
        episode = self.session_state[session_id]
        answer = extract_text(action)
        details = action.incomplete_details
        usage = action.usage
        if (
            action.status == "incomplete"
            and details is not None
            and details.reason == "max_output_tokens"
            and not answer
            and (usage is None or usage.output_tokens == 0)
        ):
            return None, 0.0, False, True, {
                "sample_id": episode.sample_id,
                "turn_rewards": list(episode.rewards),
                "turn_details": list(episode.details),
                "truncation_reason": "context_exhausted",
            }
        observation, reward, terminated, info = episode.step(answer)
        return observation, reward, terminated, False, info


if __name__ == "__main__":
    IFGymServer.run_webserver()
