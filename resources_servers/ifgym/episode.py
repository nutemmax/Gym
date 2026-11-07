# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an original IFGym conversation and record each deterministic check."""

from ifgym.data.conversation import ConversationInternalRepr
from ifgym.instructions.instructions_registry import INSTRUCTION_DICT, history_instruction_kwargs


def initial_user_text(params: dict) -> str:
    messages = params["input"]
    if isinstance(messages, str):
        return messages
    for item in reversed(messages):
        if item.get("role") == "user":
            content = item["content"]
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content if part.get("type") == "input_text")
    raise ValueError("initial input has no user message")


class Episode:
    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise ValueError("episode must be an IFGym conversation object")
        conversation = ConversationInternalRepr.from_dict(data)
        if not conversation.n_turns:
            raise ValueError("episode must have at least one turn")
        views = conversation.turn_views()
        for given, actual in zip(data["turns"], views):
            if (
                given.get("turn_idx") != actual["turn_idx"]
                or given.get("active_constraints") != actual["active_constraints"]
            ):
                raise ValueError(f"active constraints do not match events on turn {actual['turn_idx']}")
            if not isinstance(actual["rendered_prompt"], str) or not actual["rendered_prompt"].strip():
                raise ValueError("each turn requires a non-empty rendered_prompt")
            for constraint in actual["active_constraints"]:
                cid = constraint["constraint_id"]
                if cid not in INSTRUCTION_DICT:
                    raise ValueError(f"unknown constraint: {cid!r}")
                checker = INSTRUCTION_DICT[cid](cid)
                runtime_keys = history_instruction_kwargs(cid, None, []).keys()
                kwargs = constraint.get("kwargs") or {}
                required = set(checker.get_instruction_args_keys() or ()) - set(runtime_keys)
                missing = sorted(
                    key
                    for key in required
                    if key not in kwargs
                    or kwargs[key] is None
                    or isinstance(kwargs[key], (str, list, dict))
                    and not kwargs[key]
                )
                if missing:
                    raise ValueError(f"constraint {cid!r} needs explicit arguments: {missing}")
        self.sample_id = conversation.sample_id
        self.turns = views
        self.responses: list[str] = []
        self.rewards: list[float] = []
        self.details: list[dict] = []
        self.index = 0

    def step(self, response: str) -> tuple[str | None, float, bool, dict]:
        if self.index >= len(self.turns):
            raise ValueError("episode has already terminated")
        turn = self.turns[self.index]
        checked = passed = 0
        results = []
        for constraint in turn["active_constraints"]:
            cid = constraint["constraint_id"]
            kwargs = dict(constraint.get("kwargs") or {})
            kwargs.update(
                history_instruction_kwargs(cid, self.responses[-1] if self.responses else None, self.responses)
            )
            result = {**constraint, "effective_kwargs": kwargs, "checked": False, "passed": False}
            try:
                checker = INSTRUCTION_DICT[cid](cid)
                checker.build_description(**kwargs)
                checked += 1
                result["checked"] = True
                result["passed"] = bool(response and response.strip() and checker.check_following(response))
                passed += result["passed"]
            except Exception as error:
                result["error"] = {"type": type(error).__name__, "message": str(error)}
            results.append(result)
        reward = passed / checked if checked else 0.0
        self.responses.append(response)
        self.rewards.append(reward)
        self.details.append(
            {
                "turn_idx": turn["turn_idx"],
                "events": turn["events"],
                "prompt": turn["rendered_prompt"],
                "response": response,
                "active_constraints": turn["active_constraints"],
                "checks": results,
                "checked": checked,
                "passed": passed,
                "reward": reward,
            }
        )
        self.index += 1
        done = self.index == len(self.turns)
        info = {"sample_id": self.sample_id, "turn_rewards": list(self.rewards), "turn_details": list(self.details)}
        return (None if done else self.turns[self.index]["rendered_prompt"], reward, done, info)
