# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import PositiveInt, PrivateAttr


try:
    from nemo_gym.client import SESSION_ID_KEY
    from nemo_gym.config import ModelServerRef
    from nemo_gym.core import NeMoGymResponse
    from nemo_gym.server_runtime import BaseResourcesServerConfig
except ImportError:
    from nemo_gym.base_resources_server import BaseResourcesServerConfig
    from nemo_gym.config_types import ModelServerRef
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.server_utils import SESSION_ID_KEY

from ifgym.instructions.instructions_registry import (
    INSTRUCTION_DICT,
    get_instruction_metadata,
    history_instruction_kwargs,
)
from resources_servers.gymnasium import EnvResetRequest, EnvResetResponse, GymnasiumServer, extract_text
from resources_servers.ifgym.episode import Episode, initial_user_text
from resources_servers.ifgym.judge import JudgeAdapter, JudgeResult, UnresolvedGradeError


CONTEXT_WINDOW_EXCEEDED_METADATA_KEY = "nemo_gym_context_window_exceeded"
MAX_CLOSED_SESSIONS = 64
MAX_CLOSED_RECEIPTS = 128


@dataclass(frozen=True)
class _StepFailure:
    code: str
    message: str


class IFGymResourcesServerConfig(BaseResourcesServerConfig):
    max_total_sequence_length: PositiveInt
    judge_enabled: bool = False
    judge_model_server: Optional[ModelServerRef] = None
    judge_max_concurrency: PositiveInt = 2
    judge_timeout_seconds: PositiveInt = 30


class IFGymServer(GymnasiumServer):
    config: IFGymResourcesServerConfig

    _judge_adapter: Optional[JudgeAdapter] = PrivateAttr(default=None)
    _step_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _session_epochs: dict[str, int] = PrivateAttr(default_factory=dict)
    _pending_steps: dict[tuple[str, str], tuple[str, str, int, int, asyncio.Future, asyncio.Task]] = PrivateAttr(
        default_factory=dict
    )
    _step_receipts: dict[str, OrderedDict[str, tuple[str, str, int, tuple | _StepFailure]]] = PrivateAttr(
        default_factory=dict
    )
    _closed_sessions: OrderedDict[str, None] = PrivateAttr(default_factory=OrderedDict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close")(self.close_episode)
        return app

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        return self._step_locks.setdefault(session_id, asyncio.Lock())

    def _invalidate_session(self, session_id: str) -> None:
        self._session_epochs[session_id] = self._session_epochs.get(session_id, 0) + 1
        for (pending_session, _request_id), (_fingerprint, _episode_id, _turn_index, _epoch, _future, task) in list(
            self._pending_steps.items()
        ):
            if pending_session == session_id and task is not asyncio.current_task() and not task.done():
                task.cancel()

    def _mark_session_closed(self, session_id: str) -> None:
        receipts = self._step_receipts.setdefault(session_id, OrderedDict())
        while len(receipts) > MAX_CLOSED_RECEIPTS:
            receipts.popitem(last=False)
        self._closed_sessions[session_id] = None
        self._closed_sessions.move_to_end(session_id)
        while len(self._closed_sessions) > MAX_CLOSED_SESSIONS:
            expired, _ = self._closed_sessions.popitem(last=False)
            self._step_receipts.pop(expired, None)
            self._step_locks.pop(expired, None)
            self._session_epochs.pop(expired, None)

    async def close_session(self, session_id: Optional[str]) -> None:
        if session_id is None:
            await super().close_session(session_id)
            return
        self._invalidate_session(session_id)
        async with self._lock_for(session_id):
            await super().close_session(session_id)
            self._mark_session_closed(session_id)

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
        candidate = Episode(episode)
        if candidate.has_semantic_constraints:
            self._get_judge_adapter()
        old_session_id = request.session.get(SESSION_ID_KEY)
        if old_session_id is not None:
            was_open = old_session_id in self.session_state
            await self.close_session(old_session_id)
            if was_open and hasattr(self, "checkpoint_session_ended"):
                self.checkpoint_session_ended(request)
        request.session[SESSION_ID_KEY] = uuid.uuid4().hex
        return await super()._reset_endpoint(body, request)

    def _get_judge_adapter(self) -> JudgeAdapter:
        if not getattr(self.config, "judge_enabled", False):
            raise ValueError("semantic episode requires judge_enabled")
        model_ref = getattr(self.config, "judge_model_server", None)
        if model_ref is None:
            raise ValueError("semantic episode requires a configured judge model server")
        if self._judge_adapter is None:
            self._judge_adapter = JudgeAdapter(
                self.server_client,
                model_server_name=model_ref.name,
                max_concurrency=getattr(self.config, "judge_max_concurrency", 2),
                timeout_seconds=getattr(self.config, "judge_timeout_seconds", 30),
            )
        return self._judge_adapter

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[None, dict]:
        if session_id is None:
            raise ValueError("IFGym requires a session ID")
        episode_data = metadata.get("episode")
        if not isinstance(episode_data, dict):
            raise ValueError("episode metadata is required")
        episode = Episode(episode_data)
        if episode.has_semantic_constraints:
            self._get_judge_adapter()
        had_session = session_id in self.session_state
        self._invalidate_session(session_id)
        async with self._lock_for(session_id):
            self.session_state.pop(session_id, None)
            if had_session:
                self._mark_session_closed(session_id)
            self.session_state[session_id] = episode
            self._closed_sessions.pop(session_id, None)
        return None, {"supports_explicit_close": True, "supports_step_idempotency": True}

    @staticmethod
    def _action_fingerprint(action: NeMoGymResponse) -> str:
        payload = action.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cached_step(self, session_id: str, request_id: str, fingerprint: str) -> Optional[tuple]:
        receipt = self._step_receipts.get(session_id, {}).get(request_id)
        if receipt is None:
            return None
        prior_fingerprint, episode_id, _turn_index, response = receipt
        if prior_fingerprint != fingerprint:
            raise ValueError("step request ID was reused with a different response payload")
        active_episode = self.session_state.get(session_id)
        if active_episode is not None and active_episode.instance_id != episode_id:
            raise ValueError("step request ID belongs to a different IFGym episode")
        if isinstance(response, _StepFailure):
            raise UnresolvedGradeError(response.code, response.message)
        return copy.deepcopy(response)

    async def _score_semantic_constraints(self, episode: Episode, turn: dict, response: str) -> dict[str, JudgeResult]:
        adapter = self._get_judge_adapter()
        contexts = {}
        for constraint in turn["active_constraints"]:
            cid = constraint["constraint_id"]
            meta = get_instruction_metadata(cid)
            if meta.verifier_type != "llm_judge":
                continue
            if not response or not response.strip():
                continue
            handle = constraint["handle"]
            row = episode._semantic_by_handle[handle]
            kwargs = dict(constraint.get("kwargs") or {})
            kwargs.update(
                history_instruction_kwargs(
                    cid,
                    episode.responses[-1] if episode.responses else None,
                    episode.responses,
                )
            )
            instruction = INSTRUCTION_DICT[cid](cid)
            canonical_instruction = instruction.build_description(**kwargs)
            effective_task = turn.get("task_text")
            if not isinstance(effective_task, str) or not effective_task.strip():
                effective_task = next(
                    (
                        previous.get("task_text")
                        for previous in reversed(episode.turns[: episode.index])
                        if isinstance(previous.get("task_text"), str) and previous["task_text"].strip()
                    ),
                    "",
                )
            cls = INSTRUCTION_DICT[cid]
            if cid.startswith("history:") and len(episode.responses) < getattr(cls, "MIN_HISTORY", 0):
                raise UnresolvedGradeError(
                    "missing_history", f"history constraint {cid!r} lacks its declared minimum history"
                )
            history = []
            if cid.startswith("history:"):
                if cid == "history:respect_prior_user_preference":
                    binding = row["preference_binding"]
                    start = binding["origin_turn"] - 1
                else:
                    start = 0 if getattr(cls, "USES_FULL_HISTORY", False) else max(0, episode.index - 1)
                for prior_index in range(start, episode.index):
                    previous = episode.turns[prior_index]
                    history.append(
                        {"turn": previous["turn_idx"], "role": "user", "content": previous["rendered_prompt"]}
                    )
                    if cid != "history:respect_prior_user_preference" and prior_index < len(episode.responses):
                        history.append(
                            {
                                "turn": previous["turn_idx"],
                                "role": "assistant",
                                "content": episode.responses[prior_index],
                            }
                        )
            contexts[handle] = {
                "target": {
                    "constraint_id": cid,
                    "handle": handle,
                    "canonical_instruction": canonical_instruction,
                    "rubric": meta.rubric,
                    "rubric_version": meta.rubric_version,
                    "rubric_sha256": row["rubric_sha256"],
                    "registry_revision": episode.metadata["non_verifiable"]["registry_revision"],
                    "parameters": dict(constraint.get("kwargs") or {}),
                    "instruction_as_shown": row["instruction_as_shown"],
                    **(
                        {"preference_binding": row["preference_binding"]}
                        if cid == "history:respect_prior_user_preference"
                        else {}
                    ),
                },
                "evidence": {
                    "current_turn": turn["turn_idx"],
                    "effective_task": effective_task,
                    "current_user_message": turn["rendered_prompt"],
                    "candidate_response": response,
                    "relevant_history": history,
                    "binding_evidence": [],
                },
            }
        if not contexts:
            return {}
        tasks = {handle: asyncio.create_task(adapter.score(context)) for handle, context in contexts.items()}
        try:
            await asyncio.gather(*tasks.values())
        except BaseException:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        return {handle: task.result() for handle, task in tasks.items()}

    async def _step_unlocked(
        self, action: NeMoGymResponse, metadata: dict, session_id: str, episode: Episode, epoch: int
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        answer = extract_text(action)
        details = action.incomplete_details
        usage = action.usage
        action_metadata = action.metadata or {}
        sequence_limit_reached = (
            usage is not None
            and usage.input_tokens is not None
            and usage.output_tokens is not None
            and usage.input_tokens + usage.output_tokens >= self.config.max_total_sequence_length
        )
        if (
            action_metadata.get(CONTEXT_WINDOW_EXCEEDED_METADATA_KEY) == "true"
            or sequence_limit_reached
            or (
                action.status == "incomplete"
                and details is not None
                and details.reason == "max_output_tokens"
                and not answer
                and (usage is None or usage.output_tokens == 0)
            )
        ):
            return (
                None,
                0.0,
                False,
                True,
                {
                    "sample_id": episode.sample_id,
                    "turn_rewards": list(episode.rewards),
                    "turn_details": list(episode.details),
                    "truncation_reason": "context_exhausted",
                },
            )

        turn = episode.turns[episode.index]
        judge_results = await self._score_semantic_constraints(episode, turn, answer)
        if self._session_epochs.get(session_id, 0) != epoch or self.session_state.get(session_id) is not episode:
            raise UnresolvedGradeError("stale_step", "episode changed while judge work was pending")
        observation, reward, terminated, info = episode.step(answer, judge_results or None)
        if action.status == "incomplete" and details is not None and details.reason == "max_output_tokens":
            return None, reward, terminated, True, {**info, "truncation_reason": "max_output_tokens"}
        return observation, reward, terminated, False, info

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        if session_id is None:
            raise ValueError("IFGym session is missing; call reset first")
        request_id = metadata.get("_ng_step_request_id") if isinstance(metadata, dict) else None
        if request_id is not None and (not isinstance(request_id, str) or not request_id or len(request_id) > 128):
            raise ValueError("_ng_step_request_id must be a non-empty string of at most 128 characters")
        fingerprint = self._action_fingerprint(action)

        cached = self._cached_step(session_id, request_id, fingerprint) if isinstance(request_id, str) else None
        if cached is not None:
            return cached
        episode = self.session_state.get(session_id)
        if episode is None:
            raise ValueError("IFGym session is missing; call reset first")
        if episode.has_semantic_constraints and not isinstance(request_id, str):
            raise UnresolvedGradeError("missing_request_id", "semantic steps require a step request ID")

        lock = self._lock_for(session_id)
        if isinstance(request_id, str):
            pending = self._pending_steps.get((session_id, request_id))
            if pending is not None:
                prior_fingerprint, prior_episode_id, prior_turn_index, prior_epoch, future, _task = pending
                if prior_fingerprint != fingerprint:
                    raise ValueError("pending step request ID was reused with a different response payload")
                if (
                    prior_episode_id != episode.instance_id
                    or prior_turn_index != episode.index
                    or prior_epoch != self._session_epochs.get(session_id, 0)
                ):
                    raise ValueError("pending step request ID is bound to a different episode turn")
                return copy.deepcopy(await asyncio.shield(future))
        if lock.locked():
            raise ValueError("a different step is already in flight for this IFGym session")
        await lock.acquire()
        try:
            cached = self._cached_step(session_id, request_id, fingerprint) if isinstance(request_id, str) else None
            if cached is not None:
                return cached
            episode = self.session_state.get(session_id)
            if episode is None:
                raise ValueError("IFGym session is missing; call reset first")
            epoch = self._session_epochs.get(session_id, 0)
            request_task = asyncio.current_task()
            future = asyncio.get_running_loop().create_future()
            if isinstance(request_id, str):
                self._pending_steps[(session_id, request_id)] = (
                    fingerprint,
                    episode.instance_id,
                    episode.index,
                    epoch,
                    future,
                    request_task,
                )
            try:
                result = await self._step_unlocked(action, metadata, session_id, episode, epoch)
                if isinstance(request_id, str):
                    self._step_receipts.setdefault(session_id, OrderedDict())[request_id] = (
                        fingerprint,
                        episode.instance_id,
                        episode.index
                        - (0 if result[3] and result[4].get("truncation_reason") == "context_exhausted" else 1),
                        copy.deepcopy(result),
                    )
                    self._step_receipts[session_id].move_to_end(request_id)
                    if not future.done():
                        future.set_result(copy.deepcopy(result))
                return result
            except asyncio.CancelledError:
                if isinstance(request_id, str) and not future.done():
                    future.cancel()
                raise
            except UnresolvedGradeError as error:
                if isinstance(request_id, str):
                    self._step_receipts.setdefault(session_id, OrderedDict())[request_id] = (
                        fingerprint,
                        episode.instance_id,
                        episode.index,
                        _StepFailure(error.code, str(error)),
                    )
                    self._step_receipts[session_id].move_to_end(request_id)
                    if not future.done():
                        future.set_exception(error)
                        future.exception()
                raise
            except BaseException as error:
                if isinstance(request_id, str) and not future.done():
                    future.set_exception(error)
                    future.exception()
                raise
            finally:
                if isinstance(request_id, str):
                    self._pending_steps.pop((session_id, request_id), None)
        finally:
            lock.release()

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[None, dict]:
        if session_id is None:
            raise ValueError("IFGym requires a session ID")
        episode_data = metadata.get("episode")
        if not isinstance(episode_data, dict):
            raise ValueError("episode metadata is required")
        episode = Episode(episode_data)
        if episode.has_semantic_constraints:
            self._get_judge_adapter()
        had_session = session_id in self.session_state
        self._invalidate_session(session_id)
        async with self._lock_for(session_id):
            self.session_state.pop(session_id, None)
            if had_session:
                self._mark_session_closed(session_id)
            self.session_state[session_id] = episode
            self._closed_sessions.pop(session_id, None)
        return None, {"supports_explicit_close": True, "supports_step_idempotency": True}
