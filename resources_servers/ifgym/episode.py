# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run IFGym episodes with deterministic and asynchronous judge results."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from ifgym.data.conversation import ConversationInternalRepr, _rule_from_dict
from ifgym.instructions.instructions_registry import (
    INSTRUCTION_DICT,
    get_instruction_metadata,
    get_registry_revision,
    history_instruction_kwargs,
)
from ifgym.instructions.non_verifiable import get_approved_phrasing_templates
from ifgym.templates.activation_rules import ThisTurnOnly, UntilRevoked
from ifgym.templates.phrasing import PhrasingsRegistry
from ifgym.templates.single_turn_rules import (
    _capitalize_first_word_if_safe,
    _lowercase_first_word_if_safe,
)
from resources_servers.ifgym.judge import JudgeResult, UnresolvedGradeError


EMPTY_ARTIFACT_SHA256 = sha256(b"").hexdigest()


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
    def __init__(
        self,
        data: dict,
        *,
        expected_annotations_sha256: str = EMPTY_ARTIFACT_SHA256,
        expected_paraphrases_sha256: str = EMPTY_ARTIFACT_SHA256,
        paraphrases_path: str | None = None,
    ):
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
        self.expected_annotations_sha256 = expected_annotations_sha256
        self.expected_paraphrases_sha256 = expected_paraphrases_sha256
        self._phrasing_registry = None
        if paraphrases_path is not None:
            raw_phrasings = Path(paraphrases_path).read_bytes()
            actual_hash = sha256(raw_phrasings).hexdigest()
            if actual_hash != expected_paraphrases_sha256:
                raise ValueError("configured paraphrase artifact hash does not match its file")
            self._phrasing_registry = PhrasingsRegistry(paraphrases_path)
        elif expected_paraphrases_sha256 != EMPTY_ARTIFACT_SHA256:
            raise ValueError("non-empty paraphrase provenance requires a configured paraphrases_path")
        self._semantic_by_handle = self._validate_semantic_contract(data)
        self.turns = views
        self.responses: list[str] = []
        self.rewards: list[float] = []
        self.details: list[dict] = []
        self.index = 0

    @staticmethod
    def _validate_preference_binding(binding: object, turns: list[dict], introduced_at: int, handle: str) -> None:
        if not isinstance(binding, dict) or binding.get("status") != "active":
            raise ValueError(f"active preference constraint {handle!r} requires an active preference binding")

        turn_by_idx = {turn.get("turn_idx"): turn for turn in turns if isinstance(turn, dict)}
        required_text = (
            "preference_id",
            "preference_text",
            "scope",
            "origin_task_id",
            "origin_user_text",
            "origin_user_text_sha256",
            "origin_user_quote",
        )
        if any(not isinstance(binding.get(key), str) or not binding[key].strip() for key in required_text):
            raise ValueError(f"preference binding for {handle!r} has missing text fields")
        origin_turn = binding.get("origin_turn")
        if not isinstance(origin_turn, int) or isinstance(origin_turn, bool) or not 0 < origin_turn < introduced_at:
            raise ValueError(f"preference binding for {handle!r} must originate before introduction")
        origin = turn_by_idx.get(origin_turn)
        origin_text = origin.get("task_text") if origin else None
        if (
            not isinstance(origin_text, str)
            or binding["origin_user_text"] != origin_text
            or binding["origin_user_text_sha256"] != sha256(origin_text.encode("utf-8")).hexdigest()
            or binding["origin_user_quote"] not in origin_text
            or binding["origin_user_quote"] not in origin.get("rendered_prompt", "")
        ):
            raise ValueError(f"preference binding for {handle!r} does not match visible origin text")

        updates = binding.get("updates")
        if not isinstance(updates, list) or not updates:
            raise ValueError(f"preference binding for {handle!r} requires ordered preference updates")
        prior_turn = 0
        for index, update in enumerate(updates):
            if not isinstance(update, dict):
                raise ValueError(f"preference update for {handle!r} must be an object")
            event_turn = update.get("turn")
            task_id = update.get("task_id")
            action = update.get("action")
            task_hash = update.get("user_text_sha256")
            quote = update.get("visible_user_quote")
            preference_text = update.get("preference_text")
            scope = update.get("scope")
            visible_turn = turn_by_idx.get(event_turn) if isinstance(event_turn, int) else None
            visible_text = visible_turn.get("task_text") if visible_turn else None
            if (
                action not in {"set", "replace"}
                or not isinstance(event_turn, int)
                or isinstance(event_turn, bool)
                or not prior_turn < event_turn < introduced_at
                or not isinstance(task_id, str)
                or not task_id
                or not isinstance(visible_text, str)
                or task_hash != sha256(visible_text.encode("utf-8")).hexdigest()
                or not isinstance(quote, str)
                or not quote
                or quote not in visible_text
                or quote not in visible_turn.get("rendered_prompt", "")
                or not isinstance(preference_text, str)
                or not preference_text.strip()
                or not isinstance(scope, str)
                or not scope.strip()
            ):
                raise ValueError(f"preference update for {handle!r} does not match ordered visible user text")
            if index == 0 and (
                event_turn != origin_turn
                or task_id != binding["origin_task_id"]
                or visible_text != binding["origin_user_text"]
                or quote != binding["origin_user_quote"]
                or preference_text != binding["preference_text"]
                and len(updates) == 1
            ):
                raise ValueError(f"preference binding for {handle!r} has a mismatched origin event")
            prior_turn = event_turn
        last_update = updates[-1]
        if last_update["preference_text"] != binding["preference_text"] or last_update["scope"] != binding["scope"]:
            raise ValueError(f"preference binding for {handle!r} does not match its latest update")

    def _allowed_introduction_wordings(self, event: dict) -> dict[str, set[str]]:
        cid = event["constraint_id"]
        kwargs = dict(event.get("kwargs") or {})
        checker_kwargs = dict(kwargs)
        checker_kwargs.update(history_instruction_kwargs(cid, None, []))
        canonical = INSTRUCTION_DICT[cid](cid).build_description(**checker_kwargs)
        phrasing_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"prompt_to_repeat", "previous_response", "previous_responses"}
        }
        phrases = {canonical}
        for template in get_approved_phrasing_templates(cid):
            try:
                phrases.add(template.format(**phrasing_kwargs))
            except (KeyError, ValueError) as error:
                raise ValueError(f"approved phrasing template for {cid!r} has invalid event arguments") from error
        if self._phrasing_registry is not None:
            entry = self._phrasing_registry.get_entry(cid)
            if entry is not None and entry.strategy != "dynamic" and set(entry.kwargs_schema) == set(phrasing_kwargs):
                phrases.update(template.format(**phrasing_kwargs) for template in entry.phrasings)

        rule = _rule_from_dict(event["rule"])
        if isinstance(rule, (ThisTurnOnly, UntilRevoked)):
            return {phrase: {phrase} for phrase in phrases}
        prefix = rule.describe().strip()
        if not prefix:
            return {phrase: {phrase} for phrase in phrases}
        return {f"{prefix} {_lowercase_first_word_if_safe(phrase)}": {phrase} for phrase in phrases}

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
        expected_provenance = {
            "registry_revision": get_registry_revision(),
            "annotations_sha256": self.expected_annotations_sha256,
            "paraphrases_sha256": self.expected_paraphrases_sha256,
        }
        for key, expected in expected_provenance.items():
            if contract.get(key) != expected:
                raise ValueError(f"semantic metadata {key} does not match trusted runtime provenance")
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
                shown = row.get("instruction_as_shown")
                allowed_wordings = self._allowed_introduction_wordings(event)
                base_phrases = allowed_wordings.get(shown) if isinstance(shown, str) else None
                prompt = turn.get("rendered_prompt", "")
                rendered_constraints = turn.get("rendered_constraints", [])
                if (
                    not isinstance(shown, str)
                    or not shown.strip()
                    or not base_phrases
                    or not any(
                        isinstance(rendered, dict)
                        and rendered.get("constraint_id") == cid
                        and rendered.get("text") in base_phrases
                        for rendered in rendered_constraints
                    )
                    or not any(visible in prompt for visible in {shown, _lowercase_first_word_if_safe(shown), _capitalize_first_word_if_safe(shown), _capitalize_first_word_if_safe(_lowercase_first_word_if_safe(shown))})
                ):
                    raise ValueError(
                        f"semantic introduction wording for handle {handle!r} does not match its rendered registered template"
                    )
                if row.get("introduced_at") != turn.get("turn_idx"):
                    raise ValueError(f"semantic introduction turn is mismatched for handle {handle!r}")
                if cid == "history:respect_prior_user_preference":
                    self._validate_preference_binding(
                        row.get("preference_binding"),
                        episode_data["turns"],
                        row["introduced_at"],
                        handle,
                    )
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
