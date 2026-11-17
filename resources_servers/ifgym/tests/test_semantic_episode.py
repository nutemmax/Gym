# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
from ifgym.instructions.instructions_registry import INSTRUCTION_DICT, get_instruction_metadata

from resources_servers.ifgym.episode import Episode
from resources_servers.ifgym.judge import JudgeResult, UnresolvedGradeError


DATA = Path(__file__).parents[1] / "data/example.jsonl"


def semantic_episode():
    row = json.loads(DATA.read_text().splitlines()[0])["episode"]
    cid = "nonverifiable:professional_tone"
    handle = row["turns"][0]["active_constraints"][0]["handle"]
    for turn in row["turns"]:
        for event in turn["events"]:
            if event.get("handle") == handle:
                event["constraint_id"] = cid
                event["kwargs"] = {}
        for constraint in turn["active_constraints"]:
            if constraint.get("handle") == handle:
                constraint["constraint_id"] = cid
                constraint["kwargs"] = {}
    info = get_instruction_metadata(cid)
    shown = INSTRUCTION_DICT[cid](cid).build_description()
    row["metadata"]["non_verifiable"] = {
        "schema_version": 1,
        "registry_revision": "unit-test",
        "annotations_sha256": hashlib.sha256(b"annotations").hexdigest(),
        "paraphrases_sha256": hashlib.sha256(b"paraphrases").hexdigest(),
        "constraints_by_handle": {
            handle: {
                "constraint_id": cid,
                "verifier_type": "llm_judge",
                "rubric_version": info.rubric_version,
                "rubric_sha256": info.rubric_sha256,
                "instruction_as_shown": shown,
                "introduced_at": row["turns"][0]["turn_idx"],
            }
        },
    }
    return row, handle


def preference_episode():
    row = json.loads(DATA.read_text().splitlines()[0])["episode"]
    cid = "history:respect_prior_user_preference"
    handle = next(event["handle"] for event in row["turns"][1]["events"] if event.get("type") == "introduce")
    for turn in row["turns"]:
        for event in turn["events"]:
            if event.get("handle") == handle:
                event["constraint_id"] = cid
                event["kwargs"] = {}
        for constraint in turn["active_constraints"]:
            if constraint.get("handle") == handle:
                constraint["constraint_id"] = cid
                constraint["kwargs"] = {}
                constraint["requires_history"] = True

    origin_text = "I prefer free, open-source tools."
    row["turns"][0]["task_text"] = origin_text
    row["turns"][0]["rendered_prompt"] = (
        origin_text + " Include seed and do not use commas in every response until changed."
    )
    origin_hash = hashlib.sha256(origin_text.encode("utf-8")).hexdigest()
    meta = get_instruction_metadata(cid)
    shown = INSTRUCTION_DICT[cid](cid).build_description(previous_responses=[])
    row["metadata"]["non_verifiable"] = {
        "schema_version": 1,
        "registry_revision": "unit-test",
        "annotations_sha256": hashlib.sha256(b"annotations").hexdigest(),
        "paraphrases_sha256": hashlib.sha256(b"paraphrases").hexdigest(),
        "constraints_by_handle": {
            handle: {
                "constraint_id": cid,
                "verifier_type": "llm_judge",
                "rubric_version": meta.rubric_version,
                "rubric_sha256": meta.rubric_sha256,
                "instruction_as_shown": shown,
                "introduced_at": 2,
                "preference_binding": {
                    "preference_id": "preference-1",
                    "preference_text": "free, open-source tools",
                    "scope": "software recommendations",
                    "origin_turn": 1,
                    "origin_task_id": "task-1",
                    "origin_user_text": origin_text,
                    "origin_user_text_sha256": origin_hash,
                    "origin_user_quote": "I prefer free, open-source tools.",
                    "updates": [
                        {
                            "action": "set",
                            "turn": 1,
                            "task_id": "task-1",
                            "user_text_sha256": origin_hash,
                            "visible_user_quote": "I prefer free, open-source tools.",
                            "preference_text": "free, open-source tools",
                            "scope": "software recommendations",
                        }
                    ],
                    "status": "active",
                },
            }
        },
    }
    return row, handle


def test_preference_binding_is_required_and_matches_visible_prior_user_text():
    row, handle = preference_episode()
    episode = Episode(row)
    assert episode._semantic_by_handle[handle]["preference_binding"]["status"] == "active"

    row["metadata"]["non_verifiable"]["constraints_by_handle"][handle]["preference_binding"]["origin_user_quote"] = (
        "I prefer paid tools."
    )
    with pytest.raises(ValueError, match="visible origin text"):
        Episode(row)


def test_history_preference_without_binding_fails_closed():
    row, handle = preference_episode()
    del row["metadata"]["non_verifiable"]["constraints_by_handle"][handle]["preference_binding"]
    with pytest.raises(ValueError, match="requires an active preference binding"):
        Episode(row)


def test_semantic_grade_is_required_before_episode_commit():
    row, handle = semantic_episode()
    episode = Episode(row)

    with pytest.raises(UnresolvedGradeError, match="match every"):
        episode.step("A respectful answer.")

    assert episode.index == 0
    assert episode.responses == []
    assert episode.rewards == []
    assert episode.details == []

    result = JudgeResult("pass", 1, {"request_sha256": "digest"})
    _, reward, _, info = episode.step("A respectful answer.", {handle: result})
    assert episode.index == 1
    assert episode.responses == ["A respectful answer."]
    semantic = next(check for check in info["turn_details"][0]["checks"] if check["handle"] == handle)
    assert semantic["status"] == "ok"
    assert semantic["score"] == 1
    assert semantic["provenance"]["request_sha256"] == "digest"
    assert reward == 1.0


def test_semantic_contract_hash_is_checked_at_reset():
    row, _ = semantic_episode()
    next(iter(row["metadata"]["non_verifiable"]["constraints_by_handle"].values()))["rubric_sha256"] = "wrong"
    with pytest.raises(ValueError, match="rubric hash"):
        Episode(row)


def test_blank_semantic_response_fails_without_judge_result():
    row, handle = semantic_episode()
    episode = Episode(row)
    _, reward, _, info = episode.step("  ")
    check = next(item for item in info["turn_details"][0]["checks"] if item["handle"] == handle)
    assert check["status"] == "blank_response"
    assert check["score"] == 0
    assert reward == 0.0
