# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import Request

from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentRunRequest, GymnasiumRunResponse


class IFGymAgent(GymnasiumAgent):
    async def run(self, request: Request, body: GymnasiumAgentRunRequest) -> GymnasiumRunResponse:
        result = await super().run(request, body)
        rewards = result.info.get("turn_rewards", [])
        context_exhausted = result.info.get("truncation_reason") == "context_exhausted"
        if not rewards:
            if context_exhausted:
                instance_config = dict(getattr(result, "instance_config", None) or {})
                instance_config["mask_sample"] = True
                return result.model_copy(update={
                    "reward": 0.0, "instance_config": instance_config, "context_exhausted": True,
                })
            raise ValueError("IFGym rollout returned no turn rewards")
        return result.model_copy(update={
            "reward": sum(rewards) / len(rewards), "context_exhausted": context_exhausted,
        })


if __name__ == "__main__":
    IFGymAgent.run_webserver()
