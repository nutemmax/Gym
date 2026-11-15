# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import Request

try:
    from nemo_gym.core import NeMoGymResponse
except ImportError:
    from nemo_gym.openai_utils import NeMoGymResponse
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentRunRequest, GymnasiumRunResponse


def _discard_context_exhausted_turn(
    response: NeMoGymResponse, completed_turns: int
) -> tuple[NeMoGymResponse, bool]:
    """Remove the failed turn's prompt and output from the returned training transcript."""
    if completed_turns == 0:
        return response.model_copy(update={"output": []}), True

    output = response.output

    def role(item):
        return item.get("role") if isinstance(item, dict) else getattr(item, "role", None)

    last_user_index = next(
        (index for index in range(len(output) - 1, -1, -1) if role(output[index]) == "user"),
        None,
    )
    if last_user_index is None:
        # Without the prompt boundary, the preceding assistant prefix is ambiguous.
        return response.model_copy(update={"output": []}), False
    return response.model_copy(update={"output": output[:last_user_index]}), True


class IFGymAgent(GymnasiumAgent):
    async def run(self, request: Request, body: GymnasiumAgentRunRequest) -> GymnasiumRunResponse:
        result = await super().run(request, body)
        rewards = result.info.get("turn_rewards", [])
        context_exhausted = result.info.get("truncation_reason") == "context_exhausted"
        response = getattr(result, "response", None)
        prefix_preserved = True
        if context_exhausted and response is not None:
            response, prefix_preserved = _discard_context_exhausted_turn(response, len(rewards))
        updates = {"context_exhausted": context_exhausted}
        if response is not None:
            updates["response"] = response
        if context_exhausted and not prefix_preserved:
            info = dict(result.info)
            info["turn_rewards"] = []
            info["turn_details"] = []
            instance_config = dict(getattr(result, "instance_config", None) or {})
            instance_config["mask_sample"] = True
            updates.update({
                "info": info,
                "reward": 0.0,
                "instance_config": instance_config,
                "mask_sample": True,
            })
            return result.model_copy(update=updates)
        if not rewards:
            if context_exhausted:
                instance_config = dict(getattr(result, "instance_config", None) or {})
                instance_config["mask_sample"] = True
                updates.update({"reward": 0.0, "instance_config": instance_config})
                return result.model_copy(update=updates)
            raise ValueError("IFGym rollout returned no turn rewards")
        updates["reward"] = sum(rewards) / len(rewards)
        return result.model_copy(update=updates)


if __name__ == "__main__":
    IFGymAgent.run_webserver()
