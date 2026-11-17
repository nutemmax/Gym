# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run IFGym episodes with deterministic and asynchronous judge results."""

from __future__ import annotations

from typing import Mapping
from uuid import uuid4

from ifgym.data.conversation import ConversationInternalRepr
from ifgym.instructions.instructions_registry import (
    INSTRUCTION_DICT,
    get_instruction_metadata,
    history_instruction_kwargs,
)
from resources_servers.ifgym.judge import JudgeResult, UnresolvedGradeError


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
        self.instance_id = uuid4().hex
        self.sample_id = conversation.sample_id
        self.metadata = dict(conversation.metadata or {})
        self._semantic_by_handle = self._validate_semantic_contract(data)
        self.turns = views
        self.responses: list[str] = []
        self.rewards: list[float] = []
        self.details: list[dict] = []
        self.index = 0

    def _validate_semantic_contract(self, episode_data: dict) -> dict[str, dict]:
        has_semantic = any(
            event.get("type") == "introduce"
            and event.get("constraint_id") in INSTRUCTION_DICT
            and get_instruction_metadata(event["constraint_id"]).verifier_type == "llm_judge"
            for turn in episode_data["turns"]
            for event in turn.get("events", [])
        )
        if not has_semantic:
            return {}
        contract = self.metadata.get("non_verifiable")
        if not isinstance(contract, dict) or contract.get("schema_version") != 1:
            raise ValueError("semantic episode requires non_verifiable metadata schema_version 1")
        for key in ("registry_revision", "annotations_sha256", "paraphrases_sha256"):
            if not isinstance(contract.get(key), str) or not contract[key]:
                raise ValueError(f"semantic metadata requires {key}")
        rows = contract.get("constraints_by_handle")
        if not isinstance(rows, dict):
            raise ValueError("semantic episode requires constraints_by_handle metadata")
        semantic_introductions = {}
        for turn in episode_data["turns"]:
            for event in turn.get("events", []):
                if event.get("type") != "introduce":
                    continue
                cid = event.get("constraint_id")
                if cid not in INSTRUCTION_DICT or get_instruction_metadata(cid).verifier_type != "llm_judge":
                    continue
                handle = event.get("handle")
                row = rows.get(handle)
                meta = get_instruction_metadata(cid)
                if not isinstance(row, dict) or row.get("constraint_id") != cid:
                    raise ValueError(f"semantic metadata is missing or mismatched for handle {handle!r}")
                if row.get("verifier_type") != "llm_judge" or row.get("rubric_version") != meta.rubric_version:
                    raise ValueError(f"semantic rubric metadata is mismatched for handle {handle!r}")
                expected_hash = meta.rubric_sha256
                if row.get("rubric_sha256") != expected_hash:
                    raise ValueError(f"semantic rubric hash is mismatched for handle {handle!r}")
                if not isinstance(row.get("instruction_as_shown"), str) or not row["instruction_as_shown"].strip():
                    raise ValueError(f"semantic introduction wording is missing for handle {handle!r}")
                if row.get("introduced_at") != turn.get("turn_idx"):
                    raise ValueError(f"semantic introduction turn is mismatched for handle {handle!r}")
                semantic_introductions[handle] = row
        return semantic_introductions

    @property
    def has_semantic_constraints(self) -> bool:
        return bool(self._semantic_by_handle)

    def step(
        self,
        response: str,
        judge_results: Mapping[str, JudgeResult] | None = None,
    ) -> tuple[str | None, float, bool, dict]:
        if self.index >= len(self.turns):
            raise ValueError("episode has already terminated")
        turn = self.turns[self.index]
        constraints = turn["active_constraints"]
        results_by_handle = dict(judge_results or {})
        expected_handles = {
            constraint["handle"]
            for constraint in constraints
            if get_instruction_metadata(constraint["constraint_id"]).verifier_type == "llm_judge"
            and bool(response and response.strip())
        }
        if set(results_by_handle) != expected_handles:
            if expected_handles or results_by_handle:
                raise UnresolvedGradeError(
                    "missing_or_extra_grade",
                    "judge results must match every nonblank active semantic constraint handle",
                )

        checked = passed = 0
        results = []
        semantic_episode = self.has_semantic_constraints
        for constraint in constraints:
            cid = constraint["constraint_id"]
            verifier_type = get_instruction_metadata(cid).verifier_type
            kwargs = dict(constraint.get("kwargs") or {})
            kwargs.update(
                history_instruction_kwargs(cid, self.responses[-1] if self.responses else None, self.responses)
            )
            result = {**constraint, "effective_kwargs": kwargs, "checked": False, "passed": False}
            if verifier_type == "llm_judge":
                handle = constraint["handle"]
                if not response or not response.strip():
                    score = 0
                    provenance = {"status": "blank_response"}
                else:
                    grade = results_by_handle[handle]
                    score = grade.score
                    if score not in (0, 1) or grade.verdict != ("pass" if score else "fail"):
                        raise UnresolvedGradeError("invalid_grade", f"judge result for {handle!r} is invalid")
                    provenance = dict(grade.provenance)
                    provenance["status"] = "ok"
                checked += 1
                result.update(
                    verifier_type="llm_judge",
                    status=provenance["status"],
                    score=score,
                    passed=bool(score),
                    checked=True,
                    provenance=provenance,
                    rubric_version=self._semantic_by_handle[handle]["rubric_version"],
                    rubric_sha256=self._semantic_by_handle[handle]["rubric_sha256"],
                    instruction_as_shown=self._semantic_by_handle[handle]["instruction_as_shown"],
                )
                passed += int(bool(score))
                results.append(result)
                continue

            try:
                checker = INSTRUCTION_DICT[cid](cid)
                checker.build_description(**kwargs)
                checked += 1
                result["checked"] = True
                result["passed"] = bool(response and response.strip() and checker.check_following(response))
                passed += result["passed"]
            except Exception as error:
                if semantic_episode:
                    raise UnresolvedGradeError(
                        "deterministic_check_failed",
                        f"deterministic constraint {cid!r} did not produce a grade",
                    ) from error
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
