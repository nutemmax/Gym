# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pytest
from ifgym.instructions.instructions_registry import INSTRUCTION_DICT

from resources_servers.ifgym.episode import Episode
from resources_servers.ifgym.generate_examples import examples as generate_examples
from resources_servers.ifgym.verify_rollout import verify_rollout


DATA = Path(__file__).parents[1] / "data/example.jsonl"
RESPONSES = [
    ["seed helps plants", "seed, bloom", "flower helps plants", "flower All done."],
    [
        "the orchard is full of fruit and trees",
        "the orchard is full of fruit and trees",
        "THE ORCHARD HAS MANY TREES AND FRUIT",
    ],
    ["- water plants\n- water soil", "- water plants\n- water soil\n- water roots", "water helps plants grow"],
    ['{"planet":"earth"}', '{"planet":"mars"}', "Answer space is very wide"],
    ["<<River>> river flows", "<<River>> river runs clear", "river water continues to flow"],
    ["Summary a book is here Done.", "Note the author writes books Done.", "Note the setting is a quiet house Done."],
    ["blue blue water", "Green landscape is quiet", "Green water shines"],
    [
        "THE MUSIC BRINGS PEOPLE TOGETHER IN A BEAUTIFUL SONG",
        "the rhythm brings people together in a beautiful song",
        "- this instrument is called a guitar\n- this instrument is called a piano",
    ],
    ["earth and sky are wide", "earth meets sky at sunset", "<<Dawn>> earth and sky glow"],
    [
        "Begin the path is long",
        "Begin the path is quiet",
        "Arrive on the path now",
        "Arrive on the path Safe travels.",
    ],
]


def examples():
    return [json.loads(line) for line in DATA.read_text().splitlines()]


def test_examples_and_mocked_complete_rollouts():
    rows = examples()
    assert len(rows) == 10
    assert rows == generate_examples()
    ids = set()
    for row, responses in zip(rows, RESPONSES, strict=True):
        episode_data = row["episode"]
        turns = episode_data["turns"]
        assert 3 <= len(turns) <= 5
        assert row["responses_create_params"]["input"][0]["content"] == turns[0]["rendered_prompt"]
        assert row["agent_ref"] == {"type": "responses_api_agents", "name": "ifgym_agent"}
        episode = Episode(episode_data)
        for index, response in enumerate(responses):
            observation, reward, done, info = episode.step(response)
            assert done is (index == len(responses) - 1)
            assert observation == (None if done else turns[index + 1]["rendered_prompt"])
            detail = info["turn_details"][-1]
            assert detail["response"] == response
            assert detail["events"] == turns[index]["events"]
            assert detail["active_constraints"] == turns[index]["active_constraints"]
            assert [c["constraint_id"] for c in detail["checks"]] == [
                c["constraint_id"] for c in turns[index]["active_constraints"]
            ]
            assert all(c["checked"] and "error" not in c for c in detail["checks"])
            for check in detail["checks"]:
                checker = INSTRUCTION_DICT[check["constraint_id"]](check["constraint_id"])
                checker.build_description(**check["effective_kwargs"])
                assert bool(checker.check_following(response)) == check["passed"]
            ids.update(c["constraint_id"] for c in detail["checks"])
            expected = 2 / 3 if row["id"] == "ifgym-synthetic-01" and index == 1 else 1.0
            assert reward == pytest.approx(expected)
        assert info["turn_rewards"] == episode.rewards
        with pytest.raises(ValueError, match="terminated"):
            episode.step("extra")
    assert len(ids) >= 12


def test_history_constraint_receives_only_prior_assistant_responses():
    row = examples()[6]
    episode = Episode(row["episode"])
    episode.step("blue blue water")
    episode.step("Green landscape is quiet")
    _, reward, _, info = episode.step("Green water shines")
    history_check = next(
        c for c in info["turn_details"][-1]["checks"] if c["constraint_id"] == "history:same_first_word_as_previous"
    )
    assert history_check["effective_kwargs"]["previous_response"] == "Green landscape is quiet"
    full_history = next(
        c for c in info["turn_details"][-1]["checks"] if c["constraint_id"] == "history:reuse_earlier_word"
    )
    assert full_history["effective_kwargs"]["previous_responses"] == ["blue blue water", "Green landscape is quiet"]
    assert reward == 1.0


def test_snapshot_mismatch_and_unknown_constraint_are_rejected():
    data = copy.deepcopy(examples()[0]["episode"])
    data["turns"][0]["active_constraints"][0]["kwargs"] = {"keywords": ["wrong"]}
    with pytest.raises(ValueError, match="active constraints"):
        Episode(data)
    data = copy.deepcopy(examples()[0]["episode"])
    data["turns"][0]["events"][0]["constraint_id"] = "missing"
    data["turns"][0]["active_constraints"][0]["constraint_id"] = "missing"
    with pytest.raises(ValueError, match="unknown constraint"):
        Episode(data)

    data = copy.deepcopy(examples()[0]["episode"])
    data["turns"][0]["events"][0]["kwargs"] = {}
    for turn in data["turns"][:2]:
        turn["active_constraints"][0]["kwargs"] = {}
    with pytest.raises(ValueError, match="explicit arguments"):
        Episode(data)


def test_blank_response_fails_all_applicable_checks():
    episode = Episode(examples()[0]["episode"])
    _, reward, _, info = episode.step("  ")
    assert reward == 0
    assert info["turn_details"][0]["checked"] == 2
    assert all(not check["passed"] for check in info["turn_details"][0]["checks"])


def test_checker_build_errors_are_skipped_and_check_errors_count_as_failures(monkeypatch):
    class BuildError:
        def __init__(self, _constraint_id):
            pass

        def get_instruction_args_keys(self):
            return ["keywords"]

        def build_description(self, **kwargs):
            raise ValueError("cannot build")

    class CheckError:
        def __init__(self, _constraint_id):
            pass

        def get_instruction_args_keys(self):
            return []

        def build_description(self, **kwargs):
            pass

        def check_following(self, response):
            raise ValueError("cannot check")

    monkeypatch.setitem(INSTRUCTION_DICT, "keywords:existence", BuildError)
    monkeypatch.setitem(INSTRUCTION_DICT, "punctuation:no_comma", CheckError)
    episode = Episode(examples()[0]["episode"])
    _, reward, _, info = episode.step("seed")
    checks = info["turn_details"][0]["checks"]
    assert reward == 0.0
    assert [check["checked"] for check in checks] == [False, True]
    assert [check["error"]["message"] for check in checks] == ["cannot build", "cannot check"]


def test_saved_rollout_can_be_replayed_and_tampering_is_detected():
    row = examples()[0]
    episode = Episode(row["episode"])
    for response in RESPONSES[0]:
        _, _, done, info = episode.step(response)
    output = []
    for index, response in enumerate(RESPONSES[0]):
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": response}]})
        if index + 1 < len(RESPONSES[0]):
            output.append({"type": "message", "role": "user", "content": episode.turns[index + 1]["rendered_prompt"]})
    result = {
        "info": info,
        "responses_create_params": row["responses_create_params"],
        "response": {"output": output},
        "reward": sum(episode.rewards) / len(episode.rewards),
        "terminated": done,
        "truncated": False,
    }
    verify_rollout(row, result)
    altered = copy.deepcopy(result)
    altered["info"]["turn_details"][1]["checks"][0]["passed"] = False
    with pytest.raises(ValueError, match="verification differs"):
        verify_rollout(row, altered)
    altered = copy.deepcopy(result)
    altered["response"]["output"][0]["content"][0]["text"] = "different text"
    with pytest.raises(ValueError, match="assistant transcript"):
        verify_rollout(row, altered)
